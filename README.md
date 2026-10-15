# Uncertainty-Guided Sensor Placement for Soil Moisture Monitoring

A system that tells you **where to place your next soil-moisture sensor** so you learn the most from the fewest devices. It fuses a coarse, wide-area soil-moisture estimate (ERA5-Land reanalysis) with sparse ground sensors (USDA SCAN) using universal kriging, which is a Gaussian Process with a calibrated prior. It recommends new sensor sites with a calibration-aware placement criterion, and it answers questions through a natural-language interface.

**Team:** Preetisha Jain (24BCT0227) · Srishti Saluja (24BCE2587) · Bhumi Tulsyan (24BCE0673) — VIT Vellore

---

## Contribution

Satellite and reanalysis downscaling treats ground stations as fixed inputs. Sensor-placement theory chooses locations without a coarse prior. GIS chatbots only analyse data that already exists. This project joins the three:

1. **Calibration-aware placement for coarse-prior / in-situ fusion.** Station readings are modelled as `a + b·ERA5 + GP residual`, with a and b estimated by GLS (universal kriging). Sensors are placed greedily by a mutual-information ratio in which *both* terms are universal-kriging variances, so the uncertainty of the calibration line is counted. This builds on network-design work with estimated trends (Zimmerman 2006; Zhu & Stein 2006) and applies it to fusing coarse soil-moisture products with sensors.
2. **A replicated finding: with only a few sensors, a coarse prior helps only when placement is calibration-aware.** With 5 randomly placed sensors, adding the prior made error *worse* in both years (2023: 9.04% vs 8.36%; 2024: 8.13% vs 8.02%). With calibration-aware placement, the same prior gave the lowest error of all methods (2023: 7.30%; 2024: 6.88%).
3. **A development / confirmation split.** The method was developed on 2023 data, then frozen and tested once on 2024 data that played no part in any design choice. Kernels come from the previous year, and inference uses corrected confidence intervals with a Holm correction.
4. **A deployable interface.** A map app shows coverage gaps, recommends new sites and flags readings that disagree with their neighbours. Its chatbot can only *select* pre-built calculations, so every number it reports comes from the model.

## Main result: confirmatory evaluation on 2024 (`sensor_placement_confirmatory.ipynb`, section 6)

MAE (% volumetric soil moisture) at 40 held-out stations, averaged over Apr/Jun/Aug/Oct 2024 (50 runs per month). The kernel and residual scale were learned from the same month of 2023.

| Sensors | Random, no prior | Random + prior | k-means (space) + prior | k-means (space + wetness) + prior | MI + prior | **Calibration-aware MI + prior** |
|---|---|---|---|---|---|---|
| 5  | 8.02 | 8.13 | 7.19 | 7.26 | 7.39 | **6.88** |
| 10 | 7.21 | 6.84 | 6.58 | 6.70 | 6.72 | **6.52** |
| 20 | 6.57 | 6.38 | 6.38 | 6.26 | 6.31 | **6.25** |
| 30 | 6.35 | 6.24 | 6.32 | 6.18 | 6.16 | **6.09** |

Improvement of calibration-aware MI + prior (baseline MAE − method MAE). The 95% CI comes from the corrected resampled t-test, which accounts for overlapping random splits. p-values are Holm-adjusted over 20 comparisons.

| vs | Sensors | Improvement | 95% CI (MAE points) | p (Holm) | Months better |
|---|---|---|---|---|---|
| Random, no prior | 5 | **14.2%** | [+0.22, +2.05] | 0.29 | 4/4 |
| Random, no prior | 10 | **9.6%** | [+0.18, +1.19] | 0.15 | 4/4 |
| Random + prior | 5 | 15.4% | [−0.54, +3.03] | 1.00 | 4/4 |
| k-means (space) + prior | 5 | 4.3% | [−0.59, +1.20] | 1.00 | 3/4 |
| k-means (space + wetness) + prior | 5 | 5.2% | [−0.55, +1.31] | 1.00 | 3/4 |
| MI + prior | 5 | 6.8% | [−0.40, +1.41] | 1.00 | 4/4 |

**How to read this:**
- **Best on average everywhere.** Calibration-aware MI had the lowest mean error at every budget in both the development year (2023) and the confirmatory year (2024).
- **Clear gain over the original baseline.** Its gain over random placement without a prior replicated across years: 12.8% in 2023 and 14.2% in 2024 at 5 sensors. The CI excludes zero at 5 and 10 sensors in both years.
- **Smaller gains over the strong baselines.** Against k-means and plain MI the gains are 4–7%. Their CIs include zero, so they are suggestive, not established.
- **Nothing survives the strictest correction.** No comparison remains significant after Holm correction over all 20 tests. The corrected test is deliberately conservative.

The full tables for both years are printed in the notebook and saved in `experiment_results.json`.

**Satellite vs reanalysis (development protocol only, `sensor_placement.ipynb` section 14).** With NASA SMAP (36 km) as the prior, calibration-aware MI also improved on random placement without a prior (7.58% vs 8.34% at 5 sensors). ERA5-Land (9 km) was the better prior (7.16%). This comparison used the earlier protocol (kernel learned from non-test 2023 stations) and was not repeated in the confirmatory run.

---

## Repository contents

| File | What it is |
|---|---|
| `sensor_placement_confirmatory.ipynb` | **Authoritative evaluation**: corrected method, frozen protocol, 2023 development run + 2024 confirmatory run, corrected statistics |
| `sensor_placement.ipynb` | Development record: data exploration, model variants, entropy vs MI, introduction of the ERA5 prior, SMAP comparison, first k-means tests (sections 1–16). Kept unchanged. |
| `app.py`, `requirements.txt` | Streamlit app (map, recommendations, reading checks, chatbot) |
| `experiment_results.json` | Confirmatory and development tables with CIs (read by the app's chatbot) |
| `historical_kernel.json` | June 2022 kernel and residual scale, used by the app |
| `run_metadata.json` | Library versions, retrieval dates, QC exclusions with reasons, kernels and every test set |
| `snapshots/*.csv` | Station readings + ERA5-Land values exactly as used, for each month of 2022–2024 |
| `figures/confirmatory_results.png` | Development vs confirmatory results figure |
| `soil_moisture_clean.csv`, `era5_grid.csv` | June 2023 station data and map grid for the app |

### Where every number comes from

| Result | Source |
|---|---|
| Main tables above | `sensor_placement_confirmatory.ipynb` section 6 (2024) |
| Development results with the corrected code | `sensor_placement_confirmatory.ipynb` section 5 (2023) |
| SMAP vs ERA5 | `sensor_placement.ipynb` section 14 (development protocol) |
| GP model variants (Matérn, km distances, elevation) | `sensor_placement.ipynb` section 10 |

### Changes made after the code audit (October 2026)

1. GP optimiser restarts seeded. Earlier runs differed slightly: SMAP 7.60% in one run vs 7.58% in the final one.
2. ERA5-Land requests fail loudly instead of silently switching to another model, in both the notebook and the app.
3. The criterion and the predictor are both universal kriging. Previously the numerator used universal kriging and the denominator an ordinary GP, and the line was fitted by OLS.
4. The residual scale is fixed from the historical month instead of being re-estimated from the few chosen sensors.
5. Sign-test p-values on overlapping runs were replaced by corrected CIs and Holm-adjusted p-values.
6. A confirmatory evaluation was added on 2024 data, untouched during development.
7. QC: stations with a monthly mean below 1% or fewer than 20 valid days are excluded. Every exclusion is logged.

## Data

- **Ground truth:** USDA NRCS SCAN via the AWDB REST API (`SMS:-2:1`, soil moisture at 2 in / about 5 cm), monthly means, continental US, 154–170 stations per month after QC.
- **Coarse prior:** ERA5-Land `soil_moisture_0_to_7cm` (reanalysis, about 9 km) via the Open-Meteo archive API, `models=era5_land`.
- **Satellite comparison:** NASA SMAP L3 SPL3SMP (36 km) via `earthaccess`. A free Earthdata login is needed, and it is used only in `sensor_placement.ipynb`.

## How to reproduce

- **Main results:** open `sensor_placement_confirmatory.ipynb` in Google Colab and click **Runtime → Run all**. It takes about 30–40 minutes and needs no login. Results are deterministic.
- **Development record and SMAP:** `sensor_placement.ipynb`. It takes about 60 minutes and needs a NASA Earthdata login in section 14.

## How to run the app

```
pip install -r requirements.txt
streamlit run app.py
```

- **Real network mode:** uses every real sensor. It shows where sensors are missing, recommends new sites and flags readings worth checking.
- **Proof mode:** a *demonstration* on June 2023 with 40 held-out test stations and the 2022 kernel. It shows one split at a time, so its numbers differ from the paper's averages.
- **Chatbot:** works without a key (keyword matching). Add a Gemini API key in the sidebar for natural-language routing.

## Limitations

- **Scope:** one network (SCAN) in one country, with stations typically more than 100 km apart. Monthly means only.
- **Statistics:** months share most stations, so the pooled CIs treat months as independent, which may understate uncertainty. With a conservative overlap correction and 20 comparisons, no single comparison survives Holm adjustment.
- **Gains:** the gains over k-means and plain MI are small (4–7% at 5 sensors) and not statistically established. Beyond about 20 sensors, all strategies converge.
- **QC threshold:** the 1% threshold removes about 4 stations per month, mostly in the dry West (Utah and California). Some may be genuinely very dry soil rather than faulty sensors.
- **Coarse products:** ERA5-Land is a reanalysis, not a satellite retrieval. SMAP is coarse (36 km), weaker under dense vegetation, and was compared only under the development protocol.

## Key references

- A. Krause, A. Singh, C. Guestrin, "Near-optimal sensor placements in Gaussian processes," *JMLR* 9, 2008.
- D. L. Zimmerman, "Optimal network design for spatial prediction, covariance parameter estimation, and empirical prediction," *Environmetrics* 17, 2006.
- Z. Zhu, M. L. Stein, "Spatial sampling design for prediction with estimated parameters," *JABES* 11, 2006.
- C. Nadeau, Y. Bengio, "Inference for the generalization error," *Machine Learning* 52, 2003.
- G. L. Schaefer, M. H. Cosh, T. J. Jackson, "The USDA NRCS Soil Climate Analysis Network (SCAN)," *J. Atmos. Oceanic Technol.* 24, 2007.
- J. Muñoz-Sabater et al., "ERA5-Land: a state-of-the-art global reanalysis dataset for land applications," *ESSD* 13, 2021.
- D. Entekhabi et al., "The Soil Moisture Active Passive (SMAP) mission," *Proc. IEEE* 98, 2010.
