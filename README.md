GWater-FM: Groundwater Foundation Model
A foundation model for predicting groundwater dynamics in data-scarce regions, trained on 204,292 global time series from the GROW dataset and designed for transfer learning to underrepresented regions, particularly Africa.
The Problem
Africa represents 17% of the global population but only 1.73% of global groundwater monitoring data. Nigeria, home to Lagos (with ~15 million people heavily dependent on groundwater), has just 14 monitored wells in the global dataset—compared to 101,003 in the United States.
We tested whether standard ML models trained on global data can generalise to African groundwater. Results were stark:

An XGBoost model achieved R² = 0.9784 globally but collapsed to R² = -2.1277 on Africa (MAE = 28.90 m).
An LSTM achieved R² = 0.4480 globally but collapsed to R² = -0.2701 on Africa (MAE = 11.28 m).

Both models performed worse than simply predicting the mean on African data. Current approaches cannot reliably serve the continent where groundwater monitoring is scarcest and water security is most urgent.
Approach

Pretrain on >200,000 global groundwater time series paired with 36+ Earth system variables (GROW dataset, Bäthge et al., 2026).
Learn transferable, climate-aware and geology-aware representations of groundwater dynamics through masked time-step prediction.
Transfer learned representations to data-scarce African aquifer systems.
Validate against field observations from Lagos, Nigeria.

Architecture
GWater-FM is a Temporal Transformer with Static Cross-Attention, designed to fuse the two key insights from baseline experiments: geological features carry strong predictive signal (XGBoost finding) and temporal dynamics matter (LSTM finding).
Temporal Input (5 vars) ──► Embedding ──► Positional Encoding
                                          (month + year aware)
                                               │
Static Input (13 vars) ──► Static Encoder ─────┤
+ Climate Zone                                  ▼
+ Aquifer Type              Static-Temporal Cross-Attention
                            (temporal queries static context)
                                               │
                                               ▼
                                    Transformer Encoder
                                    (self-attention layers)
                                               │
                                    ┌──── Mode Switch ────┐
                               [Pretrain]            [Finetune]
                              Masked Pred.          Direct Pred.
                                   │                      │
                            + Physics Constraints   + Physics Constraints
Key architectural components:

Static Feature Encoder — Embeds porosity, permeability, elevation, drainage density, and slope as structured context tokens (not just input features), with separate learned embeddings for climate zone and aquifer type.
Static-Temporal Cross-Attention — Each timestep selectively attends to geological properties, enabling the model to learn climate–geology interactions (e.g., dry-season steps attend more to porosity; wet-season to drainage).
Temporal Positional Encoding — Sinusoidal position + month-of-year embedding + normalised year projection to capture seasonal recharge cycles and multi-year trends.
Physics-Informed Constraints — Three soft constraints with learnable weights: (1) recharge ≤ precipitation − evapotranspiration, (2) groundwater depth ≥ 0, (3) temporal smoothness to prevent unrealistic jumps.
Masked Time-Step Prediction — BERT-style self-supervised pretraining: 15% of timesteps are masked and the model predicts their groundwater depth from surrounding context.

Key Findings
Data Assessment: Africa vs. Global
MetricGlobalAfricaTotal wells204,2923,538 (1.73%)Countries5511Median groundwater depth8.4 m13.4 mMedian time series length10 years6 yearsDeclining trends22.3%28.5%Nigeria wells—14USA wells101,003—
Baseline Model Performance (Phase 2)
ModelGlobal R²Africa R²Africa MAEXGBoost (13 static features)0.9784-2.127728.90 mLSTM (5 temporal features)0.4480-0.270111.28 m
Neither static geological features alone nor temporal dynamics alone bridge the transfer gap. Both models perform worse than predicting the mean on African data, confirming the need for a foundation model approach.
Transfer Failure by African Country (XGBoost)
CountryMAE (m)SamplesRwanda (RWA)18.0911South Africa (ZAF)27.62287Namibia (NAM)34.8682
Errors scale with aridity—worst in arid Namibia, relatively better in tropical/highland Rwanda.
Feature Importance (XGBoost): Geological Variables Dominate
RankFeatureImportance1Total porosity (0–100 m)0.15162Ground elevation (MERIT)0.12093Drainage density0.11414Permeability (0–100 m)0.11155Topographic slope0.1085
Climate Zone Performance Gradient (Global Baseline)
Climate GroupMAESamplesTropical2.20 m2,362Continental3.14 m5,775Temperate4.04 m4,883Arid4.76 m3,833
Performance degrades from tropical → arid zones—the exact settings dominating much of Africa.
Foundation Model Pretraining (Phase 4)
MetricValueWells pretrained on200,754 (non-Africa)Total training steps11,484Total epochs42Loss range (normalised)0.0931 – 4.6450Final loss2.0494Pretraining objectiveMasked time-step prediction (15% mask ratio)NormalisationGlobal z-score (target mean/std from 20K well sample)Physics constraintsRecharge, depth ≥ 0, smoothness (learnable weights)Sequence stride3 (with max 20 sequences per well)ComputeGoogle Colab Pro (T4 GPU)
The model successfully learned to predict masked groundwater depths from surrounding temporal context conditioned on geological features. Loss stabilised after initial chunks and showed consistent learning across diverse hydrogeological settings.
Transfer Learning Challenges Identified

Climate mismatch — Global data is heavily temperate/North American; Africa is dominated by semi-arid/arid zones where baselines perform worst.
Aquifer differences — Global: porous dominant. Africa: more fractured/porous-fractured and karst.
Deeper water tables — African median 13.4 m vs. global 8.4 m (outside much of the training distribution).
Shorter records — African series are ~40% shorter, limiting fine-tuning signal.
Catastrophic failure — Proves the issue is fundamental, not model-specific.

Why a Foundation Model?
Baselines confirm that static geology (XGBoost) or temporal dynamics (LSTM) alone are insufficient. GWater-FM addresses this through:

Climate conditioning — Embedded climate zones so the model learns regime-specific groundwater behaviours.
Geology-aware architecture — Aquifer type, porosity, and permeability treated as structured context via cross-attention, not just input features.
Temporal encoding — Multi-year sequences processed with positional + seasonal + trend encoding to capture lag effects.
Physics-informed constraints — Enforced recharge ≤ available water, non-negative depths, and smoothness to prevent physically implausible predictions.
Self-supervised pretraining — Masked time-step prediction on 200,754 global wells to build transferable representations before fine-tuning on sparse African data.

Project Roadmap

 Phase 1: GROW data exploration and African data assessment
 Phase 2: Baseline models (XGBoost + LSTM) and transfer experiments
 Phase 3: Foundation model architecture design (temporal transformer + static cross-attention + physics module)
 Phase 4: Pretraining on global dataset (200,754 non-African wells, 11,484 steps, final loss 2.05)
 Phase 5: Transfer learning to African aquifer systems (3,538 wells)
 Phase 6: Validation against Lagos field data
 Phase 7: Open-source release with documentation and inference tools

Technical Details
Model Configuration (Colab T4)
ParameterValued_model128d_static64Attention heads8Encoder layers4 self-attention + 2 cross-attentionFeedforward dim256Temporal features5 (precipitation, PET, temperature, NDVI, actual ET)Static features13 (geological, geographic, climatic)Sequence length12 years (annual resolution)Mask ratio15%Dropout0.1OptimiserAdamW (lr=1e-4, weight_decay=0.01)SchedulerOneCycleLR (cosine annealing, 10% warmup)Mixed precisionFP16 via torch.amp
Data Processing

Normalisation: Global z-score computed from 20,000-well random sample; applied consistently across all training chunks and saved for fine-tuning/inference.
Sequence generation: Sliding window (length 12, stride 3) with maximum 20 sequences per well to prevent over-representation of long-record wells.
Chunked training: 10,000 wells per chunk to fit in Colab RAM; time series loaded per-chunk via filtered parquet reads.
Africa hold-out: All 3,538 African wells (11 countries) excluded from pretraining and reserved for transfer learning evaluation.

Data

GROW dataset (primary): Bäthge et al. (2026). "A Global-Scale Time Series Dataset for Groundwater Studies within the Earth System." Scientific Data, 13:401. DOI: 10.5281/zenodo.15149480.
(204,292 wells from 55 countries with associated Earth system variables.)
Lagos validation data: Olabode & Comte (2024, 2025).

Related Repositories

lagos-aquifer-ml — ML pipeline for Lagos hydrostratigraphy
lagos-aquifer-nn-classification — Neural network ensemble for borehole classification
lagos-recharge-analysis — Recharge threshold analysis for Lagos
groundwater-analysis — Groundwater level analysis under urban abstraction pressure

Author
Oluwaseun Franklin Olabode, PhD
University of Aberdeen, School of Geosciences

Google Scholar: scholar.google.com/citations?user=LslmGiUAAAAJ
LinkedIn: linkedin.com/in/oluwaseun-franklin-olabode-phd-6baaaa1a6

License
MIT
