# run-power-pace-tool

**run-power-pace-tool** is a specialized application designed to help runners compute the optimal pace for each segment of a route, extracted from a GPX file, so they can maintain a constant power output throughout their run. This can be particularly useful for pacing strategy on hilly courses. The algorithm is based on established academic research in exercise physiology and biomechanics.

The tool features an intuitive web interface built with Streamlit, making it easy to upload tracks, adjust settings, and visualize results.

---

## Features

- **GPX File Support:** Import any standard GPX route file.
- **Segmented Pace Calculation:** Computes target pace for every segment based on constant power output (or adjusted power, see Pace-Smoothing Power Coefficient).
- **Customizable Parameters:** Adjust runner and environmental parameters via an interactive sidebar.
- **Visual Results:** Review computed paces and route details in a clear, interactive dashboard.
- **Academic Foundation:** Algorithm based on peer-reviewed research.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/run-power-pace-tool.git
cd run-power-pace-tool
```

### 2. Create a Conda environment

Make sure you have [conda](https://docs.conda.io/en/latest/) installed.

```bash
conda create -n runpowerpace python=3.13
```

### 3. Activate the environment

```bash
conda activate runpowerpace
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Usage

1. **Import a GPX File:**
   On the app's main page, use the file uploader to import your GPX track.

2. **Configure Parameters:**
   Use the sidebar to enter relevant parameters, such as:

   - Runner characteristics (weight, height, etc.)
   - Environmental factors (temperature)
   - Target time or pace

3. **Run Computation:**
   Once the GPX file and parameters are set, click the **Run Computation** button.

4. **View Results:**
   The app will display the computed target paces for each segment of your route, along with a visualization and detailed data.

---

## Academic Reference

The underlying algorithm is based on established research in running physiology and power modeling. Please see the in-app references for details and links to the relevant publications.

---

## License

This project is licensed under the MIT License.

### Commons Clause

The software is provided under the MIT License, **except that the grant of rights does not include the right to "Sell" the Software**.

“Sell” means practicing any or all of the rights granted to you under the License to provide to third parties, for a fee or other consideration (including hosting or consulting services), a product or service whose value derives, entirely or substantially, from the functionality of the Software.

All other terms of the MIT License remain in effect.

---

## Contributions

Contributions and feedback are welcome! Please open an issue or pull request.

---

## Contact

For questions or support, please open an issue or contact me if you know me.

---

Enjoy optimal pacing with **run-power-pace-tool**!
