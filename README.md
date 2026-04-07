# GWater-FM: Groundwater Foundation Model

A foundation model for predicting groundwater dynamics in data-scarce 
regions, trained on 204,292 global time series from the GROW dataset 
and designed for transfer learning to underrepresented regions, 
particularly Africa.

## The Problem

Africa represents 17% of the global population but only 1.73% of 
global groundwater monitoring data. Nigeria, home to Lagos with 
15 million people dependent on groundwater, has just 14 monitored 
wells in the global dataset compared to 101,003 in the United States.

We tested whether standard ML models trained on global data can 
predict African groundwater. An XGBoost model achieves R² = 0.98 
globally but collapses to R² = -2.13 on Africa. An LSTM achieves 
R² = 0.45 globally but collapses to R² = -1.69 on Africa. Both 
models fail catastrophically, performing worse than predicting the 
mean. Current ML approaches cannot serve the continent where 
groundwater monitoring is most scarce and water security most urgent.

## Approach

- Pretrain on 200,000+ global groundwater time series paired with 
  36 Earth system variables (GROW dataset, Baethge et al., 2026)
- Learn transferable, climate-aware and geology-aware representations 
  of groundwater dynamics
- Transfer learned representations to data-scarce African aquifer 
  systems where monitoring infrastructure barely exists
- Validate against field observations from Lagos, Nigeria

## Key Findings

### Data Assessment: Africa vs Global

| Metric | Global | Africa |
|--------|--------|--------|
| Total wells | 204,292 | 3,538 (1.73%) |
| Countries | 55 | 11 |
| Median groundwater depth | 8.4 m | 13.4 m |
| Median time series length | 10 years | 6 years |
| Declining trends | 22.3% | 28.5% |
| Nigeria wells | — | 14 |
| USA wells | 101,003 | — |

### Baseline Model Performance

| Model | Global R² | Africa R² | Africa MAE |
|-------|-----------|-----------|------------|
| XGBoost (13 static + lagged features) | 0.9784 | -2.1277 | 28.90 m |
| LSTM (5 temporal features) | 0.4480 | -1.6909 | 19.31 m |

Both models fail on Africa, confirming that neither static geological 
features alone nor temporal dynamics alone can bridge the transfer gap. 
A foundation model combining both is needed.

### Transfer Failure by African Country

| Country | MAE (XGBoost) | Samples |
|---------|---------------|---------|
| Rwanda | 18.09 m | 11 |
| South Africa | 27.62 m | 287 |
| Namibia | 34.86 m | 82 |

Failure scales with aridity: Namibia (arid desert) shows the worst 
error, while Rwanda (tropical highland) is least affected.

### Feature Importance: Geological Variables Dominate

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Total porosity | 0.1516 |
| 2 | Ground elevation | 0.1209 |
| 3 | Drainage density | 0.1141 |
| 4 | Permeability | 0.1115 |
| 5 | Topographic slope | 0.1085 |

Subsurface geological properties control groundwater depth more than 
atmospheric forcing at the global scale.

### Climate Zone Performance Gradient

| Climate Group | MAE | Samples |
|--------------|-----|---------|
| Tropical | 2.20 m | 2,362 |
| Continental | 3.14 m | 5,775 |
| Temperate | 4.04 m | 4,883 |
| Arid | 4.76 m | 3,833 |

Model performance degrades from tropical to arid zones — exactly the 
settings where Africa needs predictions most.

## Transfer Learning Challenges Identified

- **Climate mismatch**: Global data is dominated by temperate climates; 
  Africa is dominated by semi-arid and arid settings where the model 
  performs worst (4.76 m MAE vs 2.20 m in tropical zones)
- **Aquifer type difference**: Globally porous aquifers dominate; in 
  Africa fractured and porous/fractured aquifers are most common
- **Deeper water tables**: African median depth is 13.4 m vs 8.4 m 
  globally, pushing predictions outside the training distribution
- **Shorter records**: African time series are 40% shorter than the 
  global median, limiting fine-tuning signal
- **Catastrophic transfer failure**: Both XGBoost (R² = -2.13) and 
  LSTM (R² = -1.69) fail on Africa, proving the problem is 
  fundamental, not architecture-specific

## Why a Foundation Model?

The baseline experiments prove that neither static geological features 
(XGBoost) nor temporal dynamics (LSTM) alone can bridge the Africa 
transfer gap. GWater-FM will combine both through:

1. **Climate conditioning**: Embedding climate zone information so the 
   model learns different groundwater behaviours for different settings
2. **Geology-aware representations**: Using aquifer type, porosity, and 
   permeability as architectural inputs, not just features
3. **Temporal encoding**: Processing multi-year sequences of Earth 
   system forcing to capture lag effects and trends
4. **Physics-informed constraints**: Enforcing known relationships 
   (e.g., recharge cannot exceed precipitation) to prevent wild errors
5. **Self-supervised pretraining**: Learning groundwater dynamics by 
   predicting masked time steps, building transferable understanding 
   before fine-tuning on sparse African data

## Project Roadmap

- [x] Phase 1: GROW data exploration and African data assessment
- [x] Phase 2: Baseline models (XGBoost + LSTM) and transfer experiments
- [ ] Phase 3: Foundation model architecture design
- [ ] Phase 4: Pretraining on full global dataset (204,292 wells)
- [ ] Phase 5: Transfer learning to African aquifer systems
- [ ] Phase 6: Validation against Lagos field data
- [ ] Phase 7: Open-source release with documentation

## Data

- **GROW dataset**: Baethge et al. (2026). A Global-Scale Time Series 
  Dataset for Groundwater Studies within the Earth System. Scientific 
  Data, 13:401. https://doi.org/10.5281/zenodo.15149480
- **Lagos validation data**: Olabode & Comte (2024, 2025)

## Related Repositories

- [lagos-aquifer-ml](https://github.com/seunoutlier/lagos-aquifer-ml) — 
  ML pipeline for Lagos hydrostratigraphy
- [lagos-aquifer-nn-classification](https://github.com/seunoutlier/lagos-aquifer-nn-classification) — 
  Neural network ensemble for borehole classification
- [lagos-recharge-analysis](https://github.com/seunoutlier/lagos-recharge-analysis) — 
  Recharge threshold analysis for Lagos
- [groundwater-analysis](https://github.com/seunoutlier/groundwater-analysis) — 
  Groundwater level analysis under urban abstraction pressure

## Author

**Oluwaseun Franklin Olabode, PhD**  
University of Aberdeen, School of Geosciences

- Google Scholar: https://scholar.google.com/citations?user=LslmGiUAAAAJ
- LinkedIn: https://linkedin.com/in/oluwaseun-franklin-olabode-phd-6baaaa1a6

## License

MIT
