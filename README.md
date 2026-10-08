# Explainable Intrusion Detection for Safety Critical Facilities

Code for the TCCE 2026 paper *Explainable Intrusion Detection for Safety Critical Facilities: Reducing Operator Alert Fatigue with SHAP Guided Anomaly Scoring* (Paper 89, Springer LNNS).

## Data

HAI 21.03 from https://github.com/icsdataset/hai (the `hai-21.03` folder: plain gzipped CSV, no Git LFS needed).

```
git clone --depth 1 https://github.com/icsdataset/hai.git
pip install -r requirements.txt
```

## Files

| File | Produces |
|---|---|
| `main_pipeline.ipynb` | Detection (Table 2), localisation and consolidation of the submitted version; runs in Google Colab in about five minutes |
| `revision.py` | Every analysis added in revision: consolidation baselines, tau and sigma grid, incident purity, concurrency overlay (Table 5); attribution variants (Table 4); alert populations, missed events, onset timing; bootstrap intervals |
| `detection_sensitivity.py` | Table 3: detection sensitivity to the variance criterion and the false alarm budget |
| `make_figures.py` | Figures 1 to 8 (asserts that it reproduces the paper's numbers before drawing) |
| `results/` | The outputs of the scripts exactly as reported in the paper |

```
python revision.py hai/hai-21.03 results
python detection_sensitivity.py hai/hai-21.03 results/r0_detection_sensitivity.csv
python make_figures.py hai/hai-21.03 figures results/r11_tau_sigma_grid.csv
```

Full run: about three minutes on four CPU cores. Seeds are fixed at zero.

## Fixed settings (chosen before testing, never tuned on test data)

Windows of 60 s, stride 10 s, five statistics per sensor (395 features). PCA with the smallest k reaching 98 percent of variance on the fitting split (k = 77). Threshold from a budget of 0.5 false alerts per hour on a held out calibration split of normal data. Consolidation tau = 300 s, sigma = 0.8.

## Event matching

An event is a maximal run of consecutive attack windows (a window is an attack window if any second in it is labelled attack). For a set of incidents:

* **false incident**: contains no attack window
* **merged incident**: contains windows of two or more events
* **split event**: its alert windows fall into two or more incidents
* **one to one match**: exactly one incident covers the event and that incident covers no other event
* **span**: time from the first to the last window of an incident, plus 60 s

See `consolidate()` and `purity()` in `revision.py`.

## Result files

| File | Paper |
|---|---|
| `r0_detection_sensitivity.csv` | Table 3 |
| `r11_r12_consolidation_baselines.csv` | Table 5, left |
| `r11_stress_concurrent.csv` | Table 5, right |
| `r11_tau_sigma_grid.csv` | Fig. 7 and Sect. 5.3 |
| `r13_attribution_variants.csv` | Table 4 |
| `figure_data.json` | values behind Figs. 5 and 8 (SHAP shares, chosen events) |
| `r2_onset_localisation.csv` | Sect. 5.2, timing |
| `revision_summary.json` | populations, missed events, intervals, all remaining numbers |
