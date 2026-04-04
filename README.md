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

Traditional groundwater models are too slow, too data-hungry, and 
too reliant on oversimplified conceptualisations to inform real-time 
management decisions in rapidly urbanising megacities.

## Approach

- Pretrain on 200,000+ global groundwater time series paired with 
  36 Earth system variables (GROW dataset, Baethge et al., 2026)
- Learn transferable relationships between climate, geology, land 
  use, soil properties, and groundwater dynamics
- Transfer learned representations to data-scarce African aquifer 
  systems where monitoring infrastructure barely exists
- Validate against field observations from Lagos, Nigeria

## Key Findings (Data Assessment)

| Metric | Global | Africa |
|--------|--------|--------|
| Total wells | 204,292 | 3,538 (1.73%) |
| Countries | 55 | 11 |
| Median groundwater depth | 8.4 m | 13.4 m |
| Median time series length | 10 years | 6 years |
| Declining trends | 22.3% | 28.5% |
| Nigeria wells | - | 14 |
| USA wells | 101,003 | - |

## Transfer Learning Challenges Identified

- **Climate mismatch**: Global data is dominated by temperate climates; 
  Africa is dominated by semi-arid and arid settings
- **Aquifer type difference**: Globally porous aquifers dominate; in 
  Africa fractured and porous/fractured aquifers are most common
- **Shorter records**: African time series are 40% shorter than the 
  global median, limiting fine-tuning signal

## Project Roadmap

- [x] Phase 1: GROW data exploration and African data assessment
- [ ] Phase 2: Baseline models (XGBoost, LSTM) on global subsets
- [ ] Phase 3: Foundation model architecture design
- [ ] Phase 4: Pretraining on global data
- [ ] Phase 5: Transfer learning to African aquifer systems
- [ ] Phase 6: Validation against Lagos field data
- [ ] Phase 7: Open-source release with documentation

## Data

- **GROW dataset**: Baethge et al. (2026). A Global-Scale Time Series 
  Dataset for Groundwater Studies within the Earth System. Scientific 
  Data, 13:401. https://doi.org/10.5281/zenodo.15149480
- **Lagos validation data**: Olabode & Comte (2024, 2025)

## Related Repositories

- [lagos-aquifer-ml](https://github.com/seunoutlier/lagos-aquifer-ml) - 
  ML pipeline for Lagos hydrostratigraphy
- [lagos-aquifer-nn-classification](https://github.com/seunoutlier/lagos-aquifer-nn-classification) - 
  Neural network ensemble for borehole classification
- [lagos-recharge-analysis](https://github.com/seunoutlier/lagos-recharge-analysis) - 
  Recharge threshold analysis for Lagos

## Author

**Oluwaseun Franklin Olabode, PhD**
University of Aberdeen, School of Geosciences

- Google Scholar: https://scholar.google.com/citations?user=LslmGiUAAAAJ
- LinkedIn: https://linkedin.com/in/oluwaseun-franklin-olabode-phd-6baaaa1a6

## License

MIT
