# Uncertainty-Guided Sensor Placement for Soil Moisture Monitoring

A system that tells you **where to place your next soil-moisture sensor** so you learn the most from the fewest devices. It fuses a coarse, wide-area soil-moisture estimate with sparse ground sensors using a Gaussian Process. It recommends new sensor sites with a calibration-aware placement criterion, and it answers questions through a natural-language interface.

**Team:** Preetisha Jain (24BCT0227) · Srishti Saluja (24BCE2587) · Bhumi Tulsyan (24BCE0673) — VIT Vellore

---

## Contribution

Existing work falls into three separate lines. Satellite and reanalysis downscaling treats ground stations as fixed inputs. Sensor-placement theory chooses locations without a coarse prior. GIS chatbots only analyse data that already exists. This project joins the three:

1. **Calibration-aware placement for coarse-prior / in-situ fusion.** A coarse product (ERA5-Land reanalysis or NASA SMAP satellite data) is calibrated against the ground sensors (`station ≈ a + b·coarse`), and a GP models what remains. Sensors are placed greedily using mutual information. The placement uncertainty includes the uncertainty of the calibration line itself (universal-kriging variance; see Zimmerman 2006, Zhu & Stein 2006). We apply this to the coarse-product/sensor fusion setting for soil moisture.
2. **An empirical finding: a coarse prior helps only when the sensors are well placed.** With 5 randomly placed sensors, adding the calibrated prior makes error *worse* (8.65% vs 8.33%), because the calibration line is fitted on poorly spread points. Calibration-aware placement turns the same prior into a **12.5% error reduction** (7.29%).
3. **A strict, leakage-free evaluation.** The kernel is learned from the *previous year* (2022). The only 2023 readings ever used are those of the chosen sensors. The evaluation covers 4 months, 50 paired runs each, and sign-test p-values. Baselines include random placement, k-means space-filling, and plain MI. Two coarse priors are tested: reanalysis (ERA5-Land) and satellite (SMAP).
4. **A deployable interface.** A map app shows coverage gaps, recommends new sites and flags readings that disagree with their neighbours. It includes a chatbot that can only *select* pre-built calculations, so every number it reports comes from the model.

## Main result (notebook section 16 — strict evaluation)

Prediction error (MAE, % volumetric soil moisture) at 40 held-out stations, averaged over Apr/Jun/Aug/Oct 2023 (50 runs per month):

| Sensors | Random, no prior | Random + prior | k-means (space) + prior | k-means (space + wetness) + prior | MI + prior | **Calibration-aware MI + prior** |
|---|---|---|---|---|---|---|
| 5  | 8.33 | 8.65 | 8.05 | 7.99 | 8.04 | **7.29** |
| 10 | 7.64 | 7.32 | 7.07 | 7.24 | 7.35 | **6.96** |
| 20 | 7.08 | 6.82 | 6.84 | 6.83 | 6.76 | **6.75** |
| 30 | 6.87 | 6.65 | 6.78 | 6.70 | 6.65 | **6.61** |

Paired wins out of 200 runs for calibration-aware MI + prior (one-sided sign test):

| vs | n=5 | n=10 | n=20 | n=30 |
|---|---|---|---|---|
| Random, no prior | 160 (p<0.0001) | 150 (p<0.0001) | 134 (p<0.0001) | 132 (p<0.0001) |
| Random + prior | 139 (p<0.0001) | 126 (p=0.0001) | 108 (n.s.) | 105 (n.s.) |
| k-means (space) + prior | 130 (p<0.0001) | 117 (p=0.010) | 116 (p=0.014) | 113 (p=0.038) |
| k-means (space + wetness) + prior | 129 (p<0.0001) | 133 (p<0.0001) | 110 (n.s.) | 112 (n.s.) |
| MI + prior | 120 (p=0.003) | 119 (p=0.004) | 94 (n.s.) | 97 (n.s.) |

**Satellite vs reanalysis (section 14):** the method also works with NASA SMAP. Calibration-aware MI + SMAP beats random placement without a prior in 140/200 runs at 5 sensors (p<0.0001). The 9 km ERA5-Land reanalysis is still the better prior at continental scale compared with 36 km SMAP (7.16% vs 7.60% at 5 sensors).

---

## Repository contents

| File | What it is |
|---|---|
| `sensor_placement.ipynb` | Full pipeline with outputs: data download, GP, placement, all experiments (sections 1–16) |
| `app.py` | Streamlit app (map + recommendations + reading checks + chatbot) |
| `requirements.txt` | Python packages for the app |
| `soil_moisture_clean.csv` | Cleaned station data, June 2023 (created by the notebook's Save cell) |
| `era5_grid.csv` | Coarse ERA5-Land grid for the map (created by the app on first run) |
| `historical_kernel.json` | GP kernel learned from June 2022 (section 16), used by the app's Proof mode |
| `experiment_results.json` | Section 16 averages, shown by the app's chatbot |

### Where every number comes from

| Result | Notebook section |
|---|---|
| Main table and p-values above | 16 (strict: kernel from 2022) |
| Four-month evaluation, kernel learned from non-test stations | 13 |
| SMAP satellite vs ERA5 reanalysis | 14 |
| k-means baselines | 15 (and 16) |
| GP model variants (Matern, km distances, elevation) | 10 |

## Data

- **Ground truth:** USDA NRCS SCAN network via the AWDB REST API (`SMS:-2:1`, soil moisture at 2 in / about 5 cm), monthly averages. Continental US, about 160–169 stations per month.
- **Coarse prior 1:** ERA5-Land `soil_moisture_0_to_7cm` (reanalysis, about 9 km) via the Open-Meteo archive API.
- **Coarse prior 2:** NASA SMAP L3 radiometer soil moisture, SPL3SMP (36 km), via `earthaccess` (needs a free NASA Earthdata login).

## How to reproduce

1. Open `sensor_placement.ipynb` in Google Colab and click **Runtime → Run all**.
2. In section 14, enter a NASA Earthdata username and password (free at urs.earthdata.nasa.gov).
3. A full run takes about 60 minutes.

## How to run the app

```
pip install -r requirements.txt
streamlit run app.py
```

- **Real network mode:** uses every real sensor. It shows where sensors are missing, recommends new sites and flags readings worth checking.
- **Proof mode:** starts with a few sensors, places more one by one, and scores predictions on 40 held-out test stations, using the 2022 kernel.
- **Chatbot:** works without a key (keyword matching). Add a Gemini API key in the sidebar for natural-language routing.

## Limitations

- One network (SCAN) in one country, with stations typically more than 100 km apart.
- ERA5-Land is a reanalysis, not a satellite retrieval. SMAP is coarse (36 km) and weaker under dense vegetation.
- Gains are largest at small budgets (5–10 sensors). Beyond about 20 sensors, all strategies converge.
- Monthly averages only. Day-to-day dynamics are not modelled.

## Key references

- A. Krause, A. Singh, C. Guestrin, "Near-optimal sensor placements in Gaussian processes," *JMLR* 9, 2008.
- D. L. Zimmerman, "Optimal network design for spatial prediction, covariance parameter estimation, and empirical prediction," *Environmetrics* 17, 2006.
- Z. Zhu, M. L. Stein, "Spatial sampling design for prediction with estimated parameters," *JABES* 11, 2006.
- W. Dorigo et al., "The International Soil Moisture Network," *HESS*, 2021.
- J. Muñoz-Sabater et al., "ERA5-Land: a state-of-the-art global reanalysis dataset for land applications," *ESSD* 13, 2021.
- P. E. O'Neill et al., SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture (SPL3SMP), NSIDC.
