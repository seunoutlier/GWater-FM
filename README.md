# GWater-FM: Groundwater Foundation Model

A foundation model for predicting groundwater dynamics in data-scarce regions, trained on 204,292 global time series from the GROW dataset and designed for transfer learning to underrepresented regions, particularly Africa.

## The Problem

Africa represents 17% of the global population but only **1.73%** of global groundwater monitoring data. Nigeria, home to Lagos (with ~15 million people heavily dependent on groundwater), has just **14** monitored wells in the global dataset—compared to **101,003** in the United States.

We tested whether standard ML models trained on global data can generalize to African groundwater. Results were stark:

- An **XGBoost** model achieved **R² = 0.9784** globally but collapsed to **R² = -2.1277** on Africa (MAE = 28.90 m).
- An **LSTM** achieved **R² = 0.4480** globally but collapsed to **R² = -0.2701** (or lower in variants) on Africa (MAE ~11–19 m).

Both models performed worse than simply predicting the mean on African data. Current approaches cannot reliably serve the continent where groundwater monitoring is scarcest and water security is most urgent.

## Approach

- **Pretrain** on >200,000 global groundwater time series paired with 36+ Earth system variables (GROW dataset, Bäthge et al., 2026).
- Learn transferable, **climate-aware** and **geology-aware** representations of groundwater dynamics.
- **Transfer** representations to data-scarce African aquifer systems.
- Validate against field observations from Lagos, Nigeria.

## Key Findings from Baseline Experiments

### Data Assessment: Africa vs. Global

| Metric                        | Global      | Africa          |
|-------------------------------|-------------|-----------------|
| Total wells                   | 204,292    | 3,538 (1.73%)  |
| Countries                     | 55         | 11             |
| Median groundwater depth      | 8.4 m      | 13.4 m         |
| Median time series length     | 10 years   | 6 years        |
| Declining trends              | 22.3%      | 28.5%          |
| Nigeria wells                 | —          | 14             |
| USA wells                     | 101,003    | —              |

African wells are fewer, shallower records are shorter, water tables are deeper on average, and declining trends are more prevalent.

### Baseline Model Performance

| Model                              | Global R² | Africa R²   | Africa MAE |
|------------------------------------|-----------|-------------|------------|
| XGBoost (13 static + features)     | 0.9784    | -2.1277     | 28.90 m    |
| LSTM (5 temporal features)         | 0.4480    | -0.2701     | 11.28 m    |

Neither static geological features alone nor temporal dynamics alone bridge the transfer gap. A foundation model combining both (with climate conditioning, physics constraints, and self-supervised pretraining) is required.

### Transfer Failure by African Country (XGBoost)

| Country            | MAE (m)   | Samples |
|---------------     |-----------|---------|
| Rwanda (RWA)       | 18.09     | 11      |
| South Africa (ZAF) | 27.62     | 287     |
| Namibia (NAM)      | 34.86     | 82      |

Errors scale with aridity—worst in arid Namibia, relatively better in tropical/highland Rwanda.

### Feature Importance (XGBoost): Geological Variables Dominate

| Rank | Feature                        | Importance |
|------|--------------------------------|------------|
| 1    | Total porosity (0-100 m)       | 0.1516     |
| 2    | Ground elevation (MERIT)       | 0.1209     |
| 3    | Drainage density               | 0.1141     |
| 4    | Permeability (0-100 m)         | 0.1115     |
| 5    | Topographic slope              | 0.1085     |

Subsurface properties control groundwater depth more strongly than atmospheric forcing at global scale.

### Climate Zone Performance Gradient (Global Baseline)

Performance degrades from tropical → arid zones—the exact settings dominating much of Africa.

| Climate Group | MAE    | Samples |
|---------------|--------|---------|
| Tropical      | 2.20 m | 2,362   |
| Continental   | 3.14 m | 5,775   |
| Temperate     | 4.04 m | 4,883   |
| Arid          | 4.76 m | 3,833   |

## Transfer Learning Challenges Identified

- **Climate mismatch** — Global data is heavily temperate/North American; Africa is dominated by semi-arid/arid zones where baselines perform worst.
- **Aquifer differences** — Global: porous dominant. Africa: more fractured/porous-fractured and karst.
- **Deeper water tables** — African median 13.4 m vs. global 8.4 m (outside much of the training distribution).
- **Shorter records** — African series are ~40% shorter, limiting fine-tuning.
- **Catastrophic failure** — Proves the issue is fundamental, not model-specific.

## Why a Foundation Model?

Baselines confirm that static geology (XGBoost) or temporal dynamics (LSTM) alone are insufficient. **GWater-FM** addresses this by:

1. **Climate conditioning** — Embed climate zones so the model learns regime-specific behaviours.
2. **Geology-aware architecture** — Treat aquifer type, porosity, and permeability as structured context (via cross-attention).
3. **Temporal encoding** — Process multi-year sequences with positional + seasonal + trend encoding.
4. **Physics-informed constraints** — Enforce recharge ≤ available water, non-negative depths, and smoothness to prevent physically implausible predictions.
5. **Self-supervised pretraining** — Masked time-step prediction on global data to build robust, transferable representations before sparse African fine-tuning.

## Project Roadmap

- [x] **Phase 1**: GROW data exploration and African data assessment
- [x] **Phase 2**: Baseline models (XGBoost + LSTM) and transfer experiments
- [x] **Phase 3**: Foundation model architecture design (temporal transformer + static cross-attention + physics module)
- [ ] **Phase 4**: Pretraining on full global dataset (204,292 wells)
- [ ] **Phase 5**: Transfer learning to African aquifer systems
- [ ] **Phase 6**: Validation against Lagos field data
- [ ] **Phase 7**: Open-source release with documentation and inference tools

## Data

- **GROW dataset** (primary): Bäthge et al. (2026). "A Global-Scale Time Series Dataset for Groundwater Studies within the Earth System." *Scientific Data*. DOI: [10.5281/zenodo.15149480](https://doi.org/10.5281/zenodo.15149480).  
  (204,292 wells from 55 countries with associated Earth system variables.)

- **Lagos validation data**: Olabode & Comte (2024, 2025).

## Related Repositories

- [lagos-aquifer-ml](https://github.com/seunoutlier/lagos-aquifer-ml) — ML pipeline for Lagos hydrostratigraphy
- [lagos-aquifer-nn-classification](https://github.com/seunoutlier/lagos-aquifer-nn-classification) — Neural network ensemble for borehole classification
- [lagos-recharge-analysis](https://github.com/seunoutlier/lagos-recharge-analysis) — Recharge threshold analysis for Lagos
- [groundwater-analysis](https://github.com/seunoutlier/groundwater-analysis) — Groundwater level analysis under urban abstraction pressure

## Author

**Oluwaseun Franklin Olabode, PhD**  
University of Aberdeen, School of Geosciences  

- Google Scholar: https://scholar.google.com/citations?user=LslmGiUAAAAJ  
- LinkedIn: https://linkedin.com/in/oluwaseun-franklin-olabode-phd-6baaaa1a6

## License

MIT
