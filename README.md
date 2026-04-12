**Key architectural components:**

- **Static Feature Encoder** — Embeds porosity, permeability, elevation, drainage density, and slope as structured context tokens (not just input features), with separate learned embeddings for climate zone and aquifer type.
- **Static-Temporal Cross-Attention** — Each timestep selectively attends to geological properties, enabling the model to learn climate–geology interactions (e.g., dry-season steps attend more to porosity; wet-season to drainage).
- **Temporal Positional Encoding** — Sinusoidal position + month-of-year embedding + normalised year projection to capture seasonal recharge cycles and multi-year trends.
- **Physics-Informed Constraints** — Three soft constraints with learnable weights: (1) recharge ≤ precipitation − evapotranspiration, (2) groundwater depth ≥ 0, (3) temporal smoothness to prevent unrealistic jumps.
- **Masked Time-Step Prediction** — BERT-style self-supervised pretraining: 15% of timesteps are masked and the model predicts their groundwater depth from surrounding context.

## Key Findings

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

### Baseline Model Performance (Phase 2)

| Model                              | Global R² | Africa R²   | Africa MAE |
|------------------------------------|-----------|-------------|------------|
| XGBoost (13 static features)      | 0.9784    | -2.1277     | 28.90 m    |
| LSTM (5 temporal features)         | 0.4480    | -0.2701     | 11.28 m    |

Neither static geological features alone nor temporal dynamics alone bridge the transfer gap. Both models perform worse than predicting the mean on African data, confirming the need for a foundation model approach.

### Transfer Failure by African Country (XGBoost)

| Country            | MAE (m)   | Samples |
|--------------------|-----------|---------|
| Rwanda (RWA)       | 18.09     | 11      |
| South Africa (ZAF) | 27.62     | 287     |
| Namibia (NAM)      | 34.86     | 82      |

Errors scale with aridity—worst in arid Namibia, relatively better in tropical/highland Rwanda.

### Feature Importance (XGBoost): Geological Variables Dominate

| Rank | Feature                        | Importance |
|------|--------------------------------|------------|
| 1    | Total porosity (0–100 m)       | 0.1516     |
| 2    | Ground elevation (MERIT)       | 0.1209     |
| 3    | Drainage density               | 0.1141     |
| 4    | Permeability (0–100 m)         | 0.1115     |
| 5    | Topographic slope              | 0.1085     |

### Climate Zone Performance Gradient (Global Baseline)

| Climate Group | MAE    | Samples |
|---------------|--------|---------|
| Tropical      | 2.20 m | 2,362   |
| Continental   | 3.14 m | 5,775   |
| Temperate     | 4.04 m | 4,883   |
| Arid          | 4.76 m | 3,833   |

Performance degrades from tropical → arid zones—the exact settings dominating much of Africa.

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

The model successfully learned to predict masked groundwater depths from surrounding temporal context conditioned on geological features. Loss stabilised after initial chunks and showed consistent learning across diverse hydrogeological settings.

## Transfer Learning Challenges Identified

- **Climate mismatch** — Global data is heavily temperate/North American; Africa is dominated by semi-arid/arid zones where baselines perform worst.
- **Aquifer differences** — Global: porous dominant. Africa: more fractured/porous-fractured and karst.
- **Deeper water tables** — African median 13.4 m vs. global 8.4 m (outside much of the training distribution).
- **Shorter records** — African series are ~40% shorter, limiting fine-tuning signal.
- **Catastrophic failure** — Proves the issue is fundamental, not model-specific.

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
- [ ] **Phase 5**: Transfer learning to African aquifer systems (3,538 wells)
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
| Scheduler              | OneCycleLR (cosine annealing, 10% warmup) |
| Mixed precision        | FP16 via torch.amp |

### Data Processing

- **Normalisation**: Global z-score computed from 20,000-well random sample; applied consistently across all training chunks and saved for fine-tuning/inference.
- **Sequence generation**: Sliding window (length 12, stride 3) with maximum 20 sequences per well to prevent over-representation of long-record wells.
- **Chunked training**: 10,000 wells per chunk to fit in Colab RAM; time series loaded per-chunk via filtered parquet reads.
- **Africa hold-out**: All 3,538 African wells (11 countries) excluded from pretraining and reserved for transfer learning evaluation.

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
