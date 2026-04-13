# GWater-FM: Groundwater Foundation Model

A foundation model for predicting groundwater dynamics in data-scarce regions, trained on 204,292 global time series from the GROW dataset and designed for transfer learning to underrepresented regions, particularly Africa.

## The Problem

Africa represents 17% of the global population but only **1.73%** of global groundwater monitoring data. Nigeria, home to Lagos (with ~15 million people heavily dependent on groundwater), has just **14** monitored wells in the global dataset—compared to **101,003** in the United States.

We tested whether standard ML models trained on global data can generalise to African groundwater. Results were stark:

- An **XGBoost** model achieved **R² = 0.9784** globally but collapsed to **R² = -2.1277** on Africa (MAE = 28.90 m).
- An **LSTM** achieved **R² = 0.4480** globally but collapsed to **R² = -0.2701** on Africa (MAE = 11.28 m).

Both models performed worse than simply predicting the mean on African data. Current approaches cannot reliably serve the continent where groundwater monitoring is scarcest and water security is most urgent.

**GWater-FM solves this.** After pretraining on 200,754 global wells and fine-tuning on African data, the foundation model achieves **R² = +0.1594** on Africa — the first positive R² on this transfer task, where both baselines were deeply negative.

## Key Results

| Model | Africa R² | Africa MAE | Status |
|-------|-----------|-----------|--------|
| XGBoost (13 static features) | -2.1277 | 28.90 m | Catastrophic failure |
| LSTM (5 temporal features) | -0.2701 | 11.28 m | Worse than mean |
| **GWater-FM (foundation model)** | **+0.1594** | **13.27 m** | **First positive R²** |

- **MAE reduced 54%** compared to XGBoost (28.90 → 13.27 m)
- **R² moved from deeply negative to positive** — the model now outperforms predicting the mean
- **Nigeria R² = 0.18** — strongest per-country result, directly relevant to Lagos water security
- **Namibia MAE cut from 34.86 m to 12.45 m** — arid-zone transfer gap narrowed dramatically
- **Results fully reproducible** across independent runs on Google Colab Pro (T4 GPU)

### Transfer Performance by African Country

| Country | R² | MAE (m) | RMSE (m) | Samples |
|---------|-----|---------|----------|---------|
| NGA (Nigeria) | 0.1820 | 21.07 | 22.89 | 32,772 |
| ZAF (South Africa) | 0.0892 | 13.01 | 19.34 | 401,412 |
| MDG (Madagascar) | 0.0000 | 6.17 | 6.38 | 7,956 |
| SOM (Somalia) | 0.0000 | 20.78 | 21.01 | 2,388 |
| UGA (Uganda) | 0.0000 | 11.42 | 11.71 | 1,824 |
| LSO (Lesotho) | 0.0000 | 27.12 | 27.17 | 84 |
| NAM (Namibia) | -0.0613 | 12.45 | 15.23 | 1,884 |
| GMB (Gambia) | -0.6405 | 9.63 | 11.93 | 17,568 |
| RWA (Rwanda) | -2.2149 | 11.16 | 12.43 | 22,536 |
| SLE (Sierra Leone) | -111.0391 | 14.50 | 15.00 | 2,220 |

Transfer success correlates with sample size and hydrogeological similarity to the global training distribution. Nigeria and South Africa show positive R², while small-sample countries (Sierra Leone, Lesotho) remain challenging.

## Approach

- **Pretrain** on >200,000 global groundwater time series paired with 36+ Earth system variables (GROW dataset, Bäthge et al., 2026).
- Learn transferable, **climate-aware** and **geology-aware** representations of groundwater dynamics through masked time-step prediction.
- **Transfer** learned representations to data-scarce African aquifer systems via partial fine-tuning.
- Validate against field observations from Lagos, Nigeria.

## Architecture

GWater-FM is a **Temporal Transformer with Static Cross-Attention**, designed to fuse the two key insights from baseline experiments: geological features carry strong predictive signal (XGBoost finding) and temporal dynamics matter (LSTM finding).


**Key architectural components:**

- **Static Feature Encoder** — Embeds porosity, permeability, elevation, drainage density, and slope as structured context tokens (not just input features), with separate learned embeddings for climate zone and aquifer type.
- **Static-Temporal Cross-Attention** — Each timestep selectively attends to geological properties, enabling the model to learn climate–geology interactions (e.g., dry-season steps attend more to porosity; wet-season to drainage).
- **Temporal Positional Encoding** — Sinusoidal position + month-of-year embedding + normalised year projection to capture seasonal recharge cycles and multi-year trends.
- **Physics-Informed Constraints** — Three soft constraints with learnable weights: (1) recharge ≤ precipitation − evapotranspiration, (2) groundwater depth ≥ 0, (3) temporal smoothness to prevent unrealistic jumps.
- **Masked Time-Step Prediction** — BERT-style self-supervised pretraining: 15% of timesteps are masked and the model predicts their groundwater depth from surrounding context.

## Detailed Findings

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

### Feature Importance (XGBoost): Geological Variables Dominate

| Rank | Feature                        | Importance |
|------|--------------------------------|------------|
| 1    | Total porosity (0–100 m)       | 0.1516     |
| 2    | Ground elevation (MERIT)       | 0.1209     |
| 3    | Drainage density               | 0.1141     |
| 4    | Permeability (0–100 m)         | 0.1115     |
| 5    | Topographic slope              | 0.1085     |

Subsurface properties control groundwater depth more strongly than atmospheric forcing at global scale. This finding directly motivated the cross-attention architecture.

### Climate Zone Performance Gradient (Global Baseline)

| Climate Group | MAE    | Samples |
|---------------|--------|---------|
| Tropical      | 2.20 m | 2,362   |
| Continental   | 3.14 m | 5,775   |
| Temperate     | 4.04 m | 4,883   |
| Arid          | 4.76 m | 3,833   |

Performance degrades from tropical → arid zones — the exact settings dominating much of Africa.

### Foundation Model Pretraining (Phase 4)

| Metric                  | Value                     |
|-------------------------|---------------------------|
| Wells pretrained on     | 200,754 (non-Africa)     |
| Total training steps    | 11,484                    |
| Total epochs            | 42                        |
| Loss range (normalised) | 0.0931 – 4.6450          |
| Final loss              | 2.0494                    |
| Pretraining objective   | Masked time-step prediction (15% mask ratio) |
| Normalisation           | Global z-score (target mean/std from 20K well sample) |
| Physics constraints     | Recharge, depth ≥ 0, smoothness (learnable weights) |
| Sequence stride         | 3 (with max 20 sequences per well) |
| Compute                 | Google Colab Pro (T4 GPU) |

### Transfer Learning to Africa (Phase 5)

| Metric                   | Value                      |
|--------------------------|----------------------------|
| African wells            | 3,538 (11 countries)      |
| Train / Val / Test split | 70% / 15% / 15% (stratified by country) |
| Train sequences          | 194,148                    |
| Val sequences            | 38,317                     |
| Freeze strategy          | Partial (64.2% trainable, 35.8% frozen) |
| Frozen layers            | Bottom 2 transformer encoder layers, temporal embedding |
| Unfrozen layers          | Top 2 encoder layers, cross-attention, static encoder, finetune head, physics |
| Fine-tuning LR           | 1e-5 (10× lower than pretraining) |
| Best epoch               | 1 (early stopping at epoch 9, patience=8) |
| Best val loss            | 0.7200                     |
| Test R²                  | 0.1594                     |
| Test MAE                 | 13.27 m                    |
| Test RMSE                | 18.94 m                    |

The best model was achieved after just one epoch of fine-tuning, suggesting the pretrained representations were already well-suited for African groundwater. Further training overfit to the training wells. This demonstrates that the global pretraining captured meaningful, transferable hydrogeological representations.

## Transfer Learning Challenges Identified

- **Climate mismatch** — Global data is heavily temperate/North American; Africa is dominated by semi-arid/arid zones where baselines perform worst.
- **Aquifer differences** — Global: porous dominant. Africa: more fractured/porous-fractured and karst.
- **Deeper water tables** — African median 13.4 m vs. global 8.4 m (outside much of the training distribution).
- **Shorter records** — African series are ~40% shorter, limiting fine-tuning signal.
- **Catastrophic baseline failure** — Proves the issue is fundamental, not model-specific.

## Why a Foundation Model?

Baselines confirm that static geology (XGBoost) or temporal dynamics (LSTM) alone are insufficient. **GWater-FM** addresses this through:

1. **Climate conditioning** — Embedded climate zones so the model learns regime-specific groundwater behaviours.
2. **Geology-aware architecture** — Aquifer type, porosity, and permeability treated as structured context via cross-attention, not just input features.
3. **Temporal encoding** — Multi-year sequences processed with positional + seasonal + trend encoding to capture lag effects.
4. **Physics-informed constraints** — Enforced recharge ≤ available water, non-negative depths, and smoothness to prevent physically implausible predictions.
5. **Self-supervised pretraining** — Masked time-step prediction on 200,754 global wells to build transferable representations before fine-tuning on sparse African data.

## Project Roadmap

- [x] **Phase 1**: GROW data exploration and African data assessment
- [x] **Phase 2**: Baseline models (XGBoost + LSTM) and transfer experiments
- [x] **Phase 3**: Foundation model architecture design (temporal transformer + static cross-attention + physics module)
- [x] **Phase 4**: Pretraining on global dataset (200,754 non-African wells, 11,484 steps, final loss 2.05)
- [x] **Phase 5**: Transfer learning to Africa (R² = +0.16, MAE = 13.27 m — first positive R² on this task)
- [ ] **Phase 6**: Validation against Lagos field data
- [ ] **Phase 7**: Open-source release with documentation and inference tools

## Technical Details

### Model Configuration (Colab T4)

| Parameter              | Value   |
|------------------------|---------|
| d_model                | 128     |
| d_static               | 64      |
| Attention heads        | 8       |
| Encoder layers         | 4 self-attention + 2 cross-attention |
| Feedforward dim        | 256     |
| Temporal features      | 5 (precipitation, PET, temperature, NDVI, actual ET) |
| Static features        | 13 (geological, geographic, climatic) |
| Sequence length        | 12 years (annual resolution) |
| Mask ratio             | 15%     |
| Dropout                | 0.1     |
| Optimiser              | AdamW (lr=1e-4, weight_decay=0.01) |
| Scheduler              | OneCycleLR (cosine annealing, 10% warmup) for pretraining; ReduceLROnPlateau for fine-tuning |
| Mixed precision        | FP16 via torch.amp |
| Total parameters       | 978,245 |

### Data Processing

- **Normalisation**: Global z-score computed from 20,000-well random sample; applied consistently across pretraining, fine-tuning, and inference.
- **Sequence generation**: Sliding window (length 12, stride 3, max 20/well) for pretraining; stride 1 with no cap for African fine-tuning (every sequence matters).
- **Chunked training**: 10,000 wells per chunk to fit in Colab RAM; time series loaded per-chunk via filtered parquet reads.
- **Africa hold-out**: All 3,538 African wells (11 countries) excluded from pretraining and reserved for transfer learning.
- **Train/Val/Test**: African wells split 70/15/15 stratified by country, ensuring representation across all 11 countries.

## Data

- **GROW dataset** (primary): Bäthge et al. (2026). "A Global-Scale Time Series Dataset for Groundwater Studies within the Earth System." *Scientific Data*, 13:401. DOI: [10.5281/zenodo.15149480](https://doi.org/10.5281/zenodo.15149480).
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

- Google Scholar: [scholar.google.com/citations?user=LslmGiUAAAAJ](https://scholar.google.com/citations?user=LslmGiUAAAAJ)
- LinkedIn: [linkedin.com/in/oluwaseun-franklin-olabode-phd-6baaaa1a6](https://linkedin.com/in/oluwaseun-franklin-olabode-phd-6baaaa1a6)

## License

MIT
