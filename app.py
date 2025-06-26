import streamlit as st
import gpxpy
import osmnx as ox
import numpy as np, pandas as pd
import plotly.graph_objects as go
import pyproj
from plotly.express.colors import sample_colorscale
from streamlit_folium import st_folium, folium_static
import folium
from streamlit_dimensions import st_dimensions
from datetime import datetime, timedelta

# --- Constants & helpers ---

SOFT_SURFACE_FACTOR = 1.055 # Soft surface factor for grass (trail) wrt asphalt (experiment based value)
APPROX_TIME_TOLERANCE = 0.1 # Tolerance for time convergence
RUNNING_METABOLIC_POWER_TO_CYLCING_MECHANICAL_POWER = 0.22 # Rule of thumb for running metabolic power to cycling mechanical power conversion

# Define sets for each category for OSM surface classification
OSM_PAVED = {
    'paved', 'asphalt', 'concrete',
    'concrete:lanes', 'concrete:plates',
    'paving_stones', 'sett',
    'cobblestone', 'metal', 'bricks',
    'wood'
}

OSM_UNPAVED = {
    'unpaved', 'gravel', 'dirt', 'earth', 'ground',
    'compacted', 'sand', 'mud',
    'clay', 'rock', 'pebblestone',
    'grass', 'grass_paver', 'snow',
    'ice'
}

def air_density(z_m: float, T_c: float) -> float:
    # Constants (ISA troposphere)
    P0 = 101325.0         # Pa
    T0 = 288.15           # K
    L = 0.0065            # K/m
    g = 9.80665           # m/s²
    M = 0.0289644         # kg/mol
    R_star = 8.3144598    # J/mol·K
    R = R_star / M        # ≈ 287.05 J/kg·K
    
    T_k = T_c + 273.15
    
    # Pressure at altitude
    exponent = (g * M) / (R_star * L)
    P = P0 * (1 - L * z_m / T0) ** exponent
    
    # Density
    rho = P / (R * T_k)
    return rho

def estimate_frontal_area(height_m, weight_kg):
    # From Training & Racing with a Power Meter
    return max(0.1,
        0.18964 * height_m + 0.00215 * weight_kg - 0.07861
    )

def classify_osm_surface(surface_tag: str) -> int:
    """
    Classify an OSM surface tag into 0 = 'paved', 1 = 'unpaved'.
    """
    if surface_tag is None:
        return 0
    s = surface_tag.lower()
    if s in OSM_PAVED:
        return 0
    if s in OSM_UNPAVED:
        return 1
    return 0

def gpx_to_df(gpx_file, custom_surfaces=None):
    gpx = gpxpy.parse(gpx_file)
    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                points.append((p.latitude, p.longitude, p.elevation))
    df = pd.DataFrame(points, columns=['lat','lon','ele'])
    
    if custom_surfaces is None and False:
        # Download graph with simplified buffer around the GPX track
        min_lat, max_lat = df['lat'].min(), df['lat'].max()
        min_lon, max_lon = df['lon'].min(), df['lon'].max()
        
        # Add a small buffer to ensure all points are covered
        lat_buffer = (max_lat - min_lat) * 0.05
        lon_buffer = (max_lon - min_lon) * 0.05
        
        try:
            # Try to download the graph with a timeout
            G = ox.graph_from_bbox(
                (
                    min_lon - lon_buffer,
                    min_lat - lat_buffer,
                    max_lon + lon_buffer,
                    max_lat + lat_buffer,
                ),
                network_type='all',
                simplify=True,
            )
            
            # Get nearest edges for each point
            edges = ox.distance.nearest_edges(G, df['lat'].tolist(), df['lon'].tolist())
            
            # Extract surface information
            surfaces = []
            for u, v, k in zip(*edges):
                surface_tag = G.edges[u, v, k].get('surface')
                surfaces.append(classify_osm_surface(surface_tag))
        except Exception as e:
            st.warning(f"Could not fetch surface data: {str(e)}. Using default paved surface.")
            surfaces = [0] * len(df)  # Default to paved surface
    elif custom_surfaces is None:
        # Default to paved surface if no custom surfaces are provided
        surfaces = 0
    else:
        surfaces = custom_surfaces
        if len(surfaces) < len(df):
            surfaces = surfaces + [0] * (len(df) - len(surfaces))  # pad with paved surfaces if needed

    df['surface'] = surfaces
    
    return df

def compute_segments(df, seg_length=100):
    df["idx_segment"] = np.nan
    dists = (df["cum_horiz_d"] - df["cum_horiz_d"].min()).values
    i0 = 0
    label = 0
    n = len(df)
    while i0 < n-1:
        target = dists[i0] + seg_length
        i = np.searchsorted(dists, target)
        if i>=n: i = n-1
        df.loc[i0:i, "idx_segment"] = label
        i0 = i
        label += 1
    df["idx_segment"] = df["idx_segment"].ffill().astype(int)

def datetime_to_timedelta(dt):
    """Convert a datetime.time object to a timedelta."""
    return datetime.combine(datetime.today(), dt) - datetime.combine(datetime.today(), datetime.min.time())

def compute_pace_smoothing_power_coeff(grade, coeff_max_up, coeff_max_down, alpha):
    coeff_up = (np.arctan(grade * alpha) / np.arctan(0.45 * alpha) * coeff_max_up + 1) * (grade >= 0)
    coeff_down = (np.arctan(grade * alpha) / np.arctan(0.45 * alpha) * coeff_max_down + 1) * (grade < 0)
    return coeff_up + coeff_down

def fmt_pace(x): return f"{int(x)}:{int((x%1)*60):02d}"

def main_computation(
    gpx_file,
    custom_surfaces,
    segment_length: int,
    resample_gpx_points: int | None,
    weight: float,
    height: float,
    eff: float,
    temperature: float,
    target_mode: str,
    target_total_time: float | None = None,
    target_pace: float | None = None,
    pace_smoothing_power_coeff_max_up: float = 0.1,
    pace_smoothing_power_coeff_max_down: float = 0.1,
    pace_smoothing_power_alpha: float = 10.0,
):
    # Parse GPX
    df = gpx_to_df(gpx_file, custom_surfaces=custom_surfaces)

    # Build the start and end points for each segment
    tmp1 = df.rename(columns={"lat": "lat0", "lon": "lon0", "ele": "ele0"}).copy()
    tmp2 = df.rename(columns={"lat": "lat1", "lon": "lon1", "ele": "ele1"}).copy()
    df = pd.concat([tmp1.iloc[:-1].reset_index(), tmp2.iloc[1:].reset_index().drop(columns=["surface"])], axis=1)
    tmp1, tmp2 = None, None  # free memory

    # Compute horizontal distance and elevation difference
    geod = pyproj.Geod(ellps='WGS84')
    def compute_horizontal_dist(row):
        _, _, horiz_d = geod.inv(row['lon0'], row['lat0'], row['lon1'], row['lat1'])
        return horiz_d
    df['horiz_dist'] = df.apply(compute_horizontal_dist, axis=1)
    df['d_h'] = df['ele1'] - df['ele0']
    
    # Reduce the size of the DataFrame by grouping n consecutive rows
    if resample_gpx_points is not None and resample_gpx_points > 1:
        df = df.groupby(np.arange(len(df)) // resample_gpx_points).agg({
            'lat0': 'first',
            'lon0': 'first',
            'ele0': 'first',
            'lat1': 'last',
            'lon1': 'last',
            'ele1': 'last',
            'horiz_dist': 'sum',
            'd_h': 'sum',
            'surface': 'first',  # keep the first surface type
        }).reset_index(drop=True)
    
    df['grade'] = df['d_h'] / df['horiz_dist'].replace(0, np.nan)  # avoid div by zero
    df['grade'] = df['grade'].fillna(0)  # replace NaN grades with 0
    df['cum_horiz_d'] = df['horiz_dist'].cumsum() # cumulative horizontal distance
    df['cum_horiz_d_km'] = df['cum_horiz_d'] / 1000  # convert to km

    # Convert target pace to total time if needed
    if target_mode == "Pace":
        target_total_time = target_pace * df['horiz_dist'].sum() / 1000

    # Power model parameters
    rho = air_density(df["ele0"].mean(), temperature)
    CdA = estimate_frontal_area(height, weight) * 1.0  # Cd ~ 1 for upright
    m = weight
    
    def Ci(row):
        # Minetti et al. polynomial fit for energy cost of running
        i = np.clip(row["grade"], -0.45, 0.45)
        return (155.4*i**5 - 30.4*i**4 - 43.3*i**3 + 46.3*i**2 + 19.5*i + 3.6) * eff * (SOFT_SURFACE_FACTOR if row["surface"] == 1 else 1.0)

    df['Ci'] = df.apply(Ci, axis=1)
    df['bi'] = 1 / (rho * CdA * df["horiz_dist"]**3)
    df['di'] = 8 / 27 * df['bi']**3 * m**3 * df['Ci']**3 * df['horiz_dist']**3

    def total_time_and_derivative(p, return_sum: bool = True) -> tuple[np.ndarray, np.ndarray]:
        p_corrected = p * compute_pace_smoothing_power_coeff(df["grade"], pace_smoothing_power_coeff_max_up, pace_smoothing_power_coeff_max_down, pace_smoothing_power_alpha)
        
        deltas = np.sqrt(df["bi"]**2 * p_corrected**2 + df["di"])

        times_p = 1 / (np.cbrt(df["bi"] * p_corrected + deltas) + np.cbrt(df["bi"] * p_corrected - deltas)) # Solves the power equation for time

        # Derivative of time with respect to power
        derivative_p = -1 / 3 * df["bi"] * ((1 + df["bi"] * p_corrected / deltas) * (1 / np.cbrt((df["bi"] * p_corrected + deltas)**2)) + (1 - df["bi"] * p_corrected / deltas) * (1 / np.cbrt((df["bi"] * p_corrected - deltas)**2))) / (np.cbrt(df["bi"] * p_corrected + deltas) + np.cbrt(df["bi"] * p_corrected - deltas))**2
    
        if return_sum:
            return times_p.sum(), derivative_p.sum()

        return times_p, derivative_p
    
    # Initial guess for power (constant speed, no grade)
    # P = m*Ci*v + 0.5*rho*CdA*v^3
    def P(v, ci_val): return m*ci_val*v + 0.5*rho*CdA*v**3
    mean_speed = df['horiz_dist'].sum() / target_total_time # m/s
    p_init = P(mean_speed, Ci({"grade": 0., "surface": 1})) # initial guess based on mean speed

    # Find the target power that matches the target total time
    # using Newton's method
    total_time_p, derivative_p = total_time_and_derivative(p_init)
    p_current = p_init
    f_p = total_time_p - target_total_time
    i = 0
    while abs(f_p) > APPROX_TIME_TOLERANCE:
        if derivative_p == 0:
            st.write(f"Derivative is zero, cannot converge to target time. {i}")
            p_current = np.nan
            break  # avoid division by zero
        i += 1
        p_current -= f_p / derivative_p
        total_time_p, derivative_p = total_time_and_derivative(p_current)
        f_p = total_time_p - target_total_time

    # Compute the results
    df["time"], _ = total_time_and_derivative(p_current, return_sum=False)
    df["pace"] = df["time"] / 60 / (df["horiz_dist"] / 1000) # min/km pace
    df["speed"] = df["horiz_dist"] / df["time"] # m/s speed
    df["power"] = P(df["speed"], df["Ci"]) # compute power based on speed

    # Add the smooth_pace column which is the mean of the pace over idx_segment groups
    compute_segments(df, segment_length) # groups GPS points into segments of segment_length (for plotting)
    df['smooth_pace'] = df.groupby('idx_segment')['pace'].transform('mean')

    pace_min = df['smooth_pace'].min()
    pace_max = df['smooth_pace'].max()

    # Pace annotations
    df['smooth_pace_str'] = df['smooth_pace'].apply(fmt_pace)

    return df, pace_min, pace_max

def get_summary_table(
    df: pd.DataFrame,
):
    total_distance = df["horiz_dist"].sum() / 1000  # convert to km
    total_distance_str = f"{total_distance:.2f} km"
    avg_pace = df["time"].sum() / 60 / (df["horiz_dist"].sum() / 1000)  # min/km
    avg_power = (df["power"] * df["time"]).sum() / df["time"].sum()  # average power in W
    avg_equiv_cycling_power = avg_power * RUNNING_METABOLIC_POWER_TO_CYLCING_MECHANICAL_POWER  # convert to cycling equivalent power
    final_total_time = round(df["time"].sum(), 0)
    final_total_time_nice_str = f"{int(final_total_time // 3600):01d}:{int((final_total_time % 3600) // 60):02d}:{int(final_total_time % 60):02d}" # format as HH:MM:SS
    total_elevation_pos = df.loc[df['d_h'] >= 0, 'd_h'].sum()
    total_elevation_neg = df.loc[df['d_h'] < 0, 'd_h'].sum()
    paved_total_distance = df.loc[df['surface'] == 0, 'horiz_dist'].sum() / 1000  # convert to km
    paved_prct = f"{paved_total_distance / total_distance:.1%}" if total_distance > 0 else "N/A"
    unpaved_prct = f"{1 - paved_total_distance / total_distance:.1%}" if total_distance > 0 else "N/A"
    results_summary_table = pd.DataFrame({
        "Total distance": [total_distance_str],
        "Total time": [final_total_time_nice_str],
        "Total elevation ↗": [f"{total_elevation_pos:.1f} m"],
        "Total elevation ↘": [f"{-total_elevation_neg:.1f} m"],
        "Surfaces": [f"{paved_prct} paved / {unpaved_prct} unpaved"],
        "Average pace": [f"{fmt_pace(avg_pace)}/km"],
        "Average power": [f"{avg_power:.1f} W"],
        "Average cycling equivalent power*": [f"{avg_equiv_cycling_power:.1f} W"],
    }).T.reset_index()
    results_summary_table.columns = ["Metric", "Value"]

    return results_summary_table

# --- Streamlit UI ---

st.set_page_config(page_title="Power-Based Pacing Tool", layout="wide")
st.title("📈 Power-Based Pacing Tool")

if "run_btn" not in st.session_state:
    st.session_state.run_btn = False
if "input_changed" not in st.session_state:
    st.session_state.input_changed = False
if "custom_surfaces" not in st.session_state:
    st.session_state.custom_surfaces = None
if "endpoints" not in st.session_state:
    st.session_state.endpoints = []
if "is_surface_rerun" not in st.session_state:
    st.session_state.is_surface_rerun = False

def reset_custom_surfaces():
    st.session_state.custom_surfaces = None

def button_reset_custom_surfaces():
    st.session_state.custom_surfaces = None
    st.session_state.is_surface_rerun = True
    st.session_state.endpoints = []  # reset endpoints

def on_input_change():
    st.session_state.input_changed = True

# Inputs (sidebar)
st.sidebar.title("Configurations")

st.sidebar.header("Track")
gpx_file = st.sidebar.file_uploader("Upload GPX file", type=['gpx'], on_change=reset_custom_surfaces)
# Add a selection to select whether to display the Pace computation or the surface classification page
selected_page = st.sidebar.selectbox("Select page", ["Pace Computation", "Surface Classification"], index=0, key="page_selector", help="By selecting 'Surface Classification', you can manually classify the surfaces of the GPX track. The 'Pace Computation' page will compute the pace and power based on the GPX track and the provided parameters.")
segment_length = st.sidebar.number_input("Plots Segment length (m)", value=100, on_change=on_input_change, help="Length of segments for the plots. The GPX file will be segmented into groups of this length for the map and elevation profile.")
st.sidebar.markdown("---")

st.sidebar.header("Runner")
weight = st.sidebar.number_input("Weight (kg)", value=70.0, min_value=30.0, max_value=200.0, on_change=on_input_change)
height = st.sidebar.number_input("Height (m)", value=1.75, min_value=1.2, max_value=2.2, on_change=on_input_change)
eff = st.sidebar.slider("Efficiency multiplier", 0.8, 1.2, 1.0, 0.01, help="Efficiency multiplier for the power model. 1.0 corresponds to a group of 10 elite athletes practicing endurance moutain running (men age 32.6 ± 7.5 yr, body mass 61.2 ± 5.7 kg, maximal O2 consumption (VO2 max) 68.9 ± 3.8 ml/min/kg). Lower values correspond to more efficient runners, higher values to less efficient runners.")
# cadence = st.sidebar.number_input("Avg. cadence (spm)", value=164, on_change=on_input_change)
st.sidebar.markdown("---")

st.sidebar.header("Environment")
temperature = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
st.sidebar.markdown("---")

st.sidebar.header("Smoothing")
pace_smoothing_power_coeff_max_up_raw = st.sidebar.number_input(
    "Pace-smoothing Max Power coeff. uphills (%)",
    value=10., min_value=0.0, max_value=100., step=0.5,
    format="%.1f",
    on_change=on_input_change,
    help="With a coeff. of 10% the power in steep uphills will be 10% higher than on flat sections.\nFor moderate uphills, intermediate values will be used.\nSee the plot below for the result.",
)
pace_smoothing_power_coeff_max_up = pace_smoothing_power_coeff_max_up_raw / 100.0  # convert to fraction
pace_smoothing_power_coeff_max_down_raw = st.sidebar.number_input(
    "Pace-smoothing Max Power coeff. downhills (%)",
    value=10., min_value=0.0, max_value=100., step=0.5,
    format="%.1f",
    on_change=on_input_change,
    help="With a coeff. of 10% the power in steep downhills will be 10% lower than on flat sections.\nFor moderate downhills, intermediate values will be used.\nSee the plot below for the result.",
)
pace_smoothing_power_coeff_max_down = pace_smoothing_power_coeff_max_down_raw / 100.0  # convert to fraction
pace_smoothing_power_alpha = st.sidebar.slider(
    "Pace-smoothing Power coeff. curve's alpha",
    value=10.0, min_value=.1, max_value=100.0, step=.1,
    on_change=on_input_change,
    help="Alpha controls the steepness of the smoothing curve. A higher value will make the smoothing more aggressive, while a lower value will make it more gradual.\nSee the plot below for the result.",
)
fig_smoothing_coeff = go.Figure()
x = np.linspace(-0.45, 0.45, 1000)
y = compute_pace_smoothing_power_coeff(x, pace_smoothing_power_coeff_max_up, pace_smoothing_power_coeff_max_down, pace_smoothing_power_alpha)
fig_smoothing_coeff.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Smoothing Coeff.'))
fig_smoothing_coeff.update_layout(
    title="Pace-Smoothing Power Coefficient",
    xaxis_title="Grade",
    yaxis_title=None, # "Power Coefficient",
    xaxis=dict(tickformat=".1%"),  # format as percentage
    yaxis=dict(tickformat=".1%", title_standoff=8),
    margin=dict(l=0, r=0, t=28, b=0, pad=0),
    height=200,
)
st.sidebar.plotly_chart(fig_smoothing_coeff, use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("Target")
target_mode = st.sidebar.select_slider("Target", ["Total time", "Pace"], label_visibility="hidden")
if target_mode == "Total time":
    target_total_time_dt = st.sidebar.time_input("Target time", value=datetime.now().replace(hour=1, minute=0, second=0, microsecond=0), step=timedelta(minutes=1), on_change=on_input_change)
    target_total_time = datetime_to_timedelta(target_total_time_dt).total_seconds()  # convert to seconds
else:
    col1, col2 = st.sidebar.columns(2)
    mins = col1.number_input("Min", min_value=0, max_value=59, value=4, format="%02d")
    secs = col2.number_input("Sec", min_value=0, max_value=59, value=0, format="%02d")
    target_pace = mins * 60 + secs

if st.session_state.input_changed:
    st.session_state.run_btn = True
    st.session_state.input_changed = False

run_btn = st.sidebar.button("Run Computation")

dims = st_dimensions(key="main")

should_run = run_btn or st.session_state.run_btn

if gpx_file and (should_run or selected_page == "Surface Classification"):
    df, pace_min, pace_max = main_computation(
        gpx_file=gpx_file,
        custom_surfaces=st.session_state.custom_surfaces,
        segment_length=segment_length,
        resample_gpx_points=None,
        weight=weight,
        height=height,
        eff=eff,
        temperature=temperature,
        target_mode=target_mode,
        target_total_time=target_total_time if target_mode == "Total time" else None,
        target_pace=target_pace if target_mode == "Pace" else None,
        pace_smoothing_power_coeff_max_up=pace_smoothing_power_coeff_max_up,
        pace_smoothing_power_coeff_max_down=pace_smoothing_power_coeff_max_down,
        pace_smoothing_power_alpha=pace_smoothing_power_alpha,
    )

    st.session_state.custom_surfaces = df['surface'].tolist()  # store the surfaces for later use

    if selected_page == "Surface Classification":
        # Map with segments colored by surface type

        m_surface = folium.Map(location=[df.lat0.mean(), df.lon0.mean()], zoom_start=13)

        fg_surface_start_point = folium.FeatureGroup(name="selection")
        fg_surface_track = folium.FeatureGroup(name="track")
        
        positions_surface = [*zip(df.lat0, df.lon0), (df.lat1.iloc[-1], df.lon1.iloc[-1])]
        surface_colors = ["#065fe4", "#069c51"]
        if df["surface"].values[0] == 1:
            surface_colors = surface_colors[::-1]
        folium.ColorLine(
            positions=positions_surface,
            colors=df['surface'].tolist(),
            colormap=surface_colors,  # green for unpaved, blue for paved
            weight=5,
            opacity=0.8,
        ).add_to(fg_surface_track)

        if len(st.session_state.endpoints) > 0:
            folium.CircleMarker(
                location=st.session_state.endpoints[0],
                radius=8,
                color="red",
                fill=True,
                fill_opacity=0.8,
                opacity=0.8,
            ).add_to(fg_surface_start_point)

        st.subheader("Surface Type Map")
        st.write("To change the surface from one GPS point to another, click on the starting point of the segment you want to change (it'll be marked in red), then click on the end point of the segment. The surface will toggle between paved and unpaved.\nTo remove the selected starting point, click on it again.")

        # Show a legend for the surface types colors
        st.markdown("""
        <style>
        .legend {
            display: flex;
            justify-content: space-around;
            margin-bottom: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border-radius: 50%;
        }
        </style>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: <paved_color>;"></div>
                <span>Paved</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: <unpaved_color>;"></div>
                <span>Unpaved</span>
            </div>
        </div>
        """.replace("<paved_color>", surface_colors[0]).replace("<unpaved_color>", surface_colors[1]), unsafe_allow_html=True)

        map_data = st_folium(
            m_surface,
            feature_group_to_add=[fg_surface_track, fg_surface_start_point],
            use_container_width=True,
            height=500,
            key="surface_map",
            returned_objects=["last_clicked"],
        )

        if not st.session_state.is_surface_rerun:
            if map_data.get("last_clicked"):
                coord = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

                # Find closest point in the DataFrame
                closest_idx = np.argmin(
                    np.sqrt((df['lat0'] - coord[0])**2 + (df['lon0'] - coord[1])**2)
                )
                closest_point = (df['lat0'].iloc[closest_idx], df['lon0'].iloc[closest_idx])
                geod = pyproj.Geod(ellps='WGS84')
                _, _, horiz_d = geod.inv(closest_point[1], closest_point[0], coord[1], coord[0])
                if horiz_d < 20:  # if the clicked point is within 20m of the closest point
                    if closest_point not in st.session_state.endpoints:
                        st.session_state.endpoints.append(closest_point)
                        
                        if len(st.session_state.endpoints) >= 2:
                            endpoints = st.session_state.endpoints[-2:]
                            start_idx = df[(df['lat0'] == endpoints[0][0]) & (df['lon0'] == endpoints[0][1])].index[0]
                            end_idx = df[(df['lat0'] == endpoints[1][0]) & (df['lon0'] == endpoints[1][1])].index[0]
                            current_suface = df['surface'].iloc[start_idx:end_idx+1].mode().min()
                            new_surface = 1 - current_suface  # toggle between paved (0) and unpaved (1)
                            st.session_state.custom_surfaces[start_idx:end_idx+1] = [new_surface] * (end_idx - start_idx + 1)
                            st.session_state.endpoints = []  # reset endpoints
                    else:
                        st.session_state.endpoints = []  # reset endpoints

                    st.session_state.is_surface_rerun = True
                    # Re-render the map
                    st.rerun()
        else:
            # If the map was rerun, we need to reset the state
            st.session_state.is_surface_rerun = False

        if st.session_state.custom_surfaces is not None and sum(st.session_state.custom_surfaces) > 0:
            st.button("Reset to all paved", on_click=button_reset_custom_surfaces, help="Reset all surfaces to paved. This will remove all custom surface classifications and revert to the default paved surface.")
    elif should_run:
        st.session_state.surfaces_map_init = {}

        results_summary_table = get_summary_table(df)

        # Summary table
        st.subheader("💯 Summary")
        st.dataframe(
            results_summary_table,
            hide_index=True,
            use_container_width=False,
            column_config={
                "Metric": st.column_config.Column("Metric", width="large"),
                "Value": st.column_config.Column("Value", width="medium"),
            }
        )
        st.write("<i>*In running power research, it is common to compute the metabolic power, i.e. the total energy expenditure of the runner (internal and external). In cycling, power meters measure the mechanical power, i.e. the power delivered to the pedals. The conversion factor of 0.22 is a rule of thumb to convert running metabolic power to cycling mechanical power, based on the assumption that running is less efficient than cycling (in the sense that more energy is consumed for non-forward motion). This is a rule of thumb and should be used only as an indication.</i>", unsafe_allow_html=True)

        # Map with segments colored by pace

        m = folium.Map(location=[df.lat0.mean(), df.lon0.mean()], zoom_start=13)

        positions = [*zip(df.lat0, df.lon0), (df.lat1.iloc[-1], df.lon1.iloc[-1])]
        folium.ColorLine(
            positions=positions,
            colors=df['smooth_pace'].tolist(),
            colormap=["red", "blue"],
            weight=5,
            opacity=0.8,
        ).add_to(m)

        # Overlay transparent markers for hover effect
        tooltips_location = zip(df[["lat0", "lat1"]].mean(axis=1), df[["lon0", "lon1"]].mean(axis=1))
        tootips_text = df.apply(lambda row: f"{row['cum_horiz_d_km']:.2f} km<br>{row['smooth_pace_str']}", axis=1).tolist()
        for (lat, lon), label in zip(tooltips_location, tootips_text):
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=None,
                fill=True,
                fill_opacity=0,  # invisible
                tooltip=label
            ).add_to(m)

        st.subheader("🏃🏼‍♂️ Map")
        folium_static(m, width=dims["width"] if dims else 1080, height=500)

        # Elevation profile with pace

        fig = go.Figure()

        # Iter over segment group
        for _, group in df.groupby('idx_segment'):
            cum_dist0 = group['cum_horiz_d_km'] - group['horiz_dist'] / 1000  # convert to km
            cum_dist1 = group['cum_horiz_d_km']
            pace = group['smooth_pace'].values[0]
            elevation0 = group['ele0']
            elevation1 = group['ele1']
            normalized_pace = (pace - pace_min) / (pace_max - pace_min)
            color = sample_colorscale("bluered_r", [normalized_pace])[0]
            pace_str = group['smooth_pace_str'].values[0]
            fig.add_trace(go.Scatter(
                x=cum_dist0.tolist() + cum_dist1[-1:].to_list(),
                y=elevation0.tolist() + elevation1[-1:].to_list(),
                text=[pace_str] * (len(cum_dist0) + 1),
                mode='lines',         # no markers
                line=dict(color=color),
                fill='tozeroy',
                fillcolor=color, # base is transparent, gradient done via fillgradient
                hovertemplate="%{x:.2f}km<br>%{y:.2f} m<br>%{text}<extra></extra>",
                showlegend=False
            ))

        fig.update_layout(
            title="Elevation Profile with Pace",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            coloraxis=dict(
                colorscale='bluered_r',
                cmin=pace_min,
                cmax=pace_max,
                colorbar=dict(title='Pace (min/km)')
            ),
            yaxis=dict(
                range=[df['ele0'].min() * 0.9, df['ele0'].max() * 1.1],  # Adjust y-axis range
            ),
            showlegend=True,
            margin=dict(l=0, r=0, t=28, b=0, pad=0),
        )

        st.subheader("🏃🏼‍♀️ Elevation Profile")
        st.plotly_chart(fig, use_container_width=True)

        # Segment summary
        
        bins = [-1, -0.2, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.2, 1]
        labels = ['extreme down', 'steep down', 'down', 'slight down', 'flat', 'slight up', 'up','steep up', 'extreme up']
        
        def agg_summary(dfgr):
            avg_pace = dfgr["time"].sum() / 60 / (dfgr["horiz_dist"].sum() / 1000)
            horiz_dist = dfgr["horiz_dist"].sum() / 1000
            avg_power = (dfgr["power"] * dfgr["time"]).sum() / dfgr["time"].sum()
            return pd.Series({
                'horiz_dist': horiz_dist,
                'pace': avg_pace,
                'power': avg_power
            })
        df['grade_cat'] = pd.cut(df["grade"], bins=bins, labels=labels)
        summary = df.groupby('grade_cat', observed=False).apply(agg_summary, include_groups=False).reset_index()
        
        def nice_grade_cat(cat):
            start_grade = bins[labels.index(cat)]
            end_grade = bins[labels.index(cat) + 1]
            return f"{cat.capitalize()} ({start_grade:.0%} to {end_grade:.0%})"
        
        st.subheader("⌚ Paces Summary")
        for _, s in summary.iterrows():
            if s["horiz_dist"] < 0.1:
                continue
            st.write(f"{nice_grade_cat(s['grade_cat'])}: {s['horiz_dist']:.2f} km @ {fmt_pace(s['pace'])}/km -> {s['power']:.1f} W")
elif gpx_file:
    st.info("Configure settings and click 'Run Computation' to analyze the GPX file.")
else:
    st.info("Upload a GPX file and configure settings to run.")

# Add empty space
st.write("")
st.write("")
st.write("")

# Sources

with st.expander("🗂️ Sources", expanded=False):
    st.markdown("""
- Minetti et al., J. Appl. Physiol., 93:1039-46, 2002 – measured energy cost of running at various slopes and provided a polynomial fit. [(paper)](https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.01177.2001)
- Pugh, J. Physiol., 1970 – studied oxygen intake in track vs. treadmill, quantified air resistance cost (~8% at 21.5 km/h). [(paper)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1348744/pdf/jphysiol01058-0278.pdf)
- Skiba PF., 2006 – “Calculation of Power Output and ... Development of the GOVSS Algorithm” – introduced a method to estimate running power from grade and pace. [(paper)](https://raw.githubusercontent.com/GoldenCheetah/GoldenCheetah/master/doc/user/govss.pdf)
- van Dijk & van Megen, The Secret of Running, 2017 (as quoted in TrainingPeaks blog) – outlined running power components (horizontal, climbing, air) and example pacing adjustments.
- DeepSearch by ChatGPT (4.5) – used to find relevant sources and references for running power estimation, compile the information and build the main algorithm (and code).
""")