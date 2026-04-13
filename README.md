# GWater-FM: Groundwater Foundation Model

A foundation model for predicting groundwater dynamics in data-scarce regions, trained on 204,292 global time series from the GROW dataset and designed for transfer learning to underrepresented regions, particularly Africa.

## The Problem

Africa represents 17% of the global population but only **1.73%** of global groundwater monitoring data. Nigeria, home to Lagos (with ~15 million people heavily dependent on groundwater), has just **14** monitored wells in the global dataset—compared to **101,003** in the United States.

We tested whether standard ML models trained on global data can generalise to African groundwater. Results were stark:

- An **XGBoost** model achieved **R² = 0.9784** globally but collapsed to **R² = -2.1277** on Africa (MAE = 28.90 m).
- An **LSTM** achieved **R² = 0.4480** globally but collapsed to **R² = -0.2701** on Africa (MAE = 11.28 m).

Both models performed worse than simply predicting the mean on African data. Current approaches cannot reliably serve the continent where groundwater monitoring is scarcest and water security is most urgent.

## Approach

- **Pretrain** on >200,000 global groundwater time series paired with 36+ Earth system variables (GROW dataset, Bäthge et al., 2026).
- Learn transferable, **climate-aware** and **geology-aware** representations of groundwater dynamics through masked time-step prediction.
- **Transfer** learned representations to data-scarce African aquifer systems.
- Validate against field observations from Lagos, Nigeria.

## Architecture

GWater-FM is a **Temporal Transformer with Static Cross-Attention**, designed to fuse the two key insights from baseline experiments: geological features carry strong predictive signal (XGBoost finding) and temporal dynamics matter (LSTM finding).
