# Final Project Report (Week 8)

## 1. Background and Research Motivation

This project is conducted in the context of quantitative option research, with a specific focus on chooser options. A chooser option allows the holder to decide at time \(T_1\) whether the contract will become a European call or a European put, while keeping the same strike \(K\) and maturity \(T_2\). In textbook settings, this product can be valued using closed-form frameworks such as Rubinstein/BSM under a set of restrictive assumptions. In practical markets, however, pricing performance is affected by changing volatility regimes, interest-rate shifts, and sentiment-related state changes. The internship therefore sets a clear target: start from a theoretically sound baseline, then build a data-driven enhancement path and eventually deliver a usable tool interface.

The work spans eight weeks and is organized as a complete pipeline rather than isolated experiments. The pipeline covers data collection, preprocessing, model replication, baseline validation, machine-learning extension, robustness diagnostics, sensitivity analysis, and toolization. The final output is not only a model comparison table but also an executable pricing prototype with API and dashboard views, allowing both research interpretation and demonstration delivery.

## 2. Problem Definition and Practical Constraints

The primary task is to estimate chooser option value in a way that is both theoretically grounded and empirically usable. A direct supervised-learning setup based on publicly available chooser transaction prices is difficult because this data is sparse and not cleanly accessible in open channels. To preserve reproducibility and keep the evaluation consistent across milestones, the project uses a documented realized-proxy target as the benchmark target. This choice introduces limitations, but it enables a stable model-development loop and transparent comparison between baseline and ML variants.

From a research perspective, the project has four practical questions. First, whether BSM/Rubinstein can be faithfully replicated as a baseline under modern data tooling. Second, whether ML enhancement can meaningfully reduce pricing error relative to baseline in out-of-sample tests. Third, whether model behavior is explainable in terms of economically meaningful drivers such as volatility, rates, and sentiment proxy. Fourth, whether the entire workflow can be wrapped into a demonstrable tool with near-real-time data refresh.

## 3. Data Pipeline and Feature Construction

The data layer integrates market series from Yahoo Finance and macro-rate series from FRED. The Yahoo block provides JPM OHLCV, VIX, and dividends; the FRED block provides rate information used in risk-free-rate related features. Data engineering is implemented with reliability in mind: missing values are handled through interpolation, outliers are treated with IQR-based logic, and all series are aligned to a trading-day index with forward filling where appropriate. To avoid pipeline fragility in cloud environments, fallback logic is included when certain FRED resources are unavailable.

After cleaning and alignment, the feature set combines traditional and regime-aware signals. Traditional signals include return and realized volatility windows, while regime-aware signals include VIX level/momentum, rate momentum, sentiment proxy, and moneyness-related variables. The implementation is maintained in `src/preprocess.py` and `src/features/feature_engineering.py`, and definitions are documented in `docs/feature_engineering.md`. This design ensures each downstream model uses the same canonical feature source, reducing experiment drift.

## 4. Baseline Modeling: BSM/Rubinstein Replication

The baseline stage reproduces chooser-option valuation through both simulation-based and analytic routes. Monte Carlo path logic is implemented to maintain conceptual consistency with stochastic pricing intuition, and Rubinstein closed-form pricing is used as the primary deterministic baseline for subsequent comparisons. This phase establishes the benchmark behavior before introducing machine-learning complexity.

Week 4 validation focuses on error measurement and regime diagnostics. With the realized-proxy target in place, baseline predictions are evaluated through MAE and RMSE, then inspected under market-state segmentation. This provides two essential outputs: a quantitative benchmark that can be reused in later weeks and a qualitative map of baseline failure modes under stressed regimes.

## 5. Machine Learning Framework and Evaluation Method

The ML phase introduces two complementary architectures. The first architecture predicts volatility and feeds it back into BSM-style pricing, preserving financial-structure interpretability. The second architecture performs end-to-end supervised pricing directly on engineered features. This dual design is intentional: it allows a controlled comparison between structurally constrained ML and purely predictive ML.

To avoid look-ahead bias, all model development uses strict chronological splitting with a 70/15/15 train/validation/test configuration. Hyperparameter optimization uses time-series cross-validation. Evaluation uses MAE, RMSE, and R², while additional diagnostics include train/val/test gap analysis, search-space transparency, and collinearity checks. In practice, this means the project evaluates not only point performance but also generalization behavior and model risk indicators.

## 6. Key Quantitative Results

The Week 6 benchmark comparison shows a clear improvement path from baseline to advanced models. The BSM baseline records MAE 34.7897 and RMSE 37.8863 on the test target. The hybrid route using XGB volatility plus BSM repricing improves this to MAE 32.5829 and RMSE 35.0129. End-to-end models further improve error in selected configurations, with the best model (`approach2_mlp`) reaching MAE 27.3385 and RMSE 29.8961, corresponding to roughly 21% improvement over baseline on both MAE and RMSE dimensions in the stored benchmark output.

Beyond point metrics, robustness diagnostics provide additional context. Train/validation/test differences indicate that the test period remains significantly harder than in-sample windows, consistent with distribution drift rather than simple memorization. Hyperparameter spaces and selected best parameters are exported (`data/reports/week6/hyperparameter_search_spaces.json`) to make model selection auditable. Correlation and VIF diagnostics are also included, with explicit treatment of deterministic redundancy between `moneyness_t` and `s_t`.

## 7. Explainability, Sensitivity, and Stress Validation

Explainability is addressed with SHAP/LIME in Week 6 and extended in Week 7. A global SHAP ranking is first generated to identify key drivers. To respond to follow-up concerns regarding conditional behavior, segmented SHAP analysis is then added along two axes: maturity buckets and moneyness buckets. The corresponding outputs are stored in `data/reports/week7/shap_by_maturity_bucket.csv` and `data/reports/week7/shap_by_moneyness_bucket.csv`. This directly addresses whether `vix_t` and `sentiment_proxy` contributions remain stable or shift across product states.

Stress testing initially uses deterministic scenarios aligned with project requirements: \(+50\%\) volatility shock, \(+200\) bps rate shock, and a combined shock. To test realism, these synthetic scenarios are then calibrated against historical event windows, specifically the 2020 crash and the 2022 hiking cycle. The comparison output (`data/reports/week7/historical_event_calibration.csv`) shows how scenario magnitudes relate to observed historical peaks, allowing explicit discussion of whether the stress envelope is conservative or insufficient under different market episodes.

## 8. Tool Completion and Deployment-Oriented Delivery

The final week shifts emphasis from model development to tool completion. A service layer (`src/tooling/pricing_tool.py`) is implemented to centralize pricing logic and dashboard data assembly. This layer loads the selected ML model, computes dual pricing outputs, estimates error margins from historical residuals, and provides trend and sensitivity tables.

On the interface side, Streamlit is upgraded from prototype to a structured pricing dashboard. The app now supports dual pricing display (BSM and ML), uncertainty bands (68% and 95%), trend plots for target versus model outputs, rolling error views, and sensitivity/performance table rendering. In parallel, the FastAPI service is extended with dual-pricing and dashboard endpoints, including `/price/dual`, `/dashboard/series`, `/dashboard/metrics`, and `/dashboard/sensitivity`, while retaining health, default config, and market-update endpoints.

For near-real-time integration, `src/data/market_updater.py` provides incremental JPM/VIX refresh and merging into raw storage, so the tool can update data without rerunning full historical collection each time. This balances practical refresh speed and reproducibility requirements.

## 9. Deliverables and Completion Status

By the end of Week 8, the repository contains all required final-stage components. The pricing tool is deployable as a prototype with runnable Streamlit and FastAPI entry points and documented commands in README. The final written synthesis is provided in this report file and can be exported to PDF. Demo recording preparation is completed via a structured script in `docs/week8_demo_video_script.md`, and presentation preparation is completed via `docs/week8_presentation_deck.md`.

The project therefore closes with complete research traceability from data to model to interface, rather than isolated notebooks or non-reproducible snapshots.

## 10. Limitations and Future Work

The first limitation is target construction: the benchmark remains a reproducible proxy rather than direct chooser transaction prices. The second limitation is sentiment representation: current sentiment is VIX-derived and does not yet integrate NLP-based news flow. The third limitation is stress modeling: deterministic shocks provide clarity but cannot fully capture joint regime transitions and path-dependent structural breaks. The fourth limitation is deployment maturity: while the tool is robust for demonstration and internal usage, production-grade concerns such as authentication, observability, and model governance should be further strengthened.

Future development should prioritize richer sentiment ingestion, walk-forward retraining and monitoring schedules, expanded stress engines, and deployment hardening with model versioning and structured logs. These steps can convert the current strong prototype into a maintainable production-facing quant service.

## 11. Conclusion

This internship project successfully translates chooser-option pricing research into a reproducible engineering pipeline and a demonstrable tool. It starts with theoretical baseline replication, quantifies baseline limitations, introduces ML enhancement under strict time-series validation, extends analysis with explainability and historical stress calibration, and finishes with a dual-pricing dashboard and API surface. The final outcome is not only an improvement in benchmark error metrics but also a clearer, more operational way to analyze and present pricing behavior under changing market regimes.

