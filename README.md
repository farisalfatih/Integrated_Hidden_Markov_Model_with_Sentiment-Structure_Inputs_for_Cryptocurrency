# Integrated Hidden Markov Model with Sentiment-Structure Inputs for Cryptocurrency

**Crypto Market Regime Prediction Pipeline** – A complete, end‑to‑end pipeline that ingests OHLCV data from Coinbase, merges Bitcoin Dominance (BTCD) and Fear & Greed Index (FGI), engineers technical features, detects hidden market regimes using a Gaussian HMM (trained **without look‑ahead bias**), constructs future regime targets, trains Random Forest and XGBoost classifiers, performs ablation studies, and generates publication‑ready figures and SHAP analyses.

---

## 📁 Project Structure

```
.
├── crypto_regime_pipeline.py   # Main pipeline script (single file)
├── data_raw/                   # Raw data (Coinbase CSVs, BTCD, FGI)
├── data_processed/             # All generated outputs
│   ├── coinbase_merged/
│   ├── coin_selection/
│   ├── btcd/
│   ├── fgi/
│   ├── master_data/
│   ├── feature_data/
│   ├── regime_data/
│   ├── model_data/
│   ├── model_results_rf/
│   ├── model_results_xgb/
│   ├── ablation_results/
│   ├── model_comparison/
│   ├── shap_results/
│   ├── publication_figures/
│   └── publication_tables/
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost hmmlearn shap joblib requests tqdm matplotlib seaborn
```

> **Note:** For GPU‑accelerated XGBoost, ensure CUDA and `xgboost` with GPU support are installed.

### 2. Prepare Data

Place your raw data in the following structure (or adjust using `--data_dir`):

```
data_raw/
├── data_coinbase_all_1h_part1/     # Part 1 of Coinbase OHLCV CSVs
├── data_coinbase_all_1h_part2/     # Part 2 of Coinbase OHLCV CSVs
├── data_btcd/
│   ├── Bitcoin_7_9_2022-9_8_2022_historical_data_coinmarketcap.csv
│   ├── Bitcoin_11_1_2025-11_20_2025_historical_data_coinmarketcap.csv
│   ├── Bitcoin_11_3_2025-5_2_2026_historical_data_coinmarketcap.csv
│   └── CoinGecko-GlobalCryptoMktCap-2026-05-02 (2).csv
```

The pipeline will automatically detect the Coinbase directories even if they are nested one level deeper (e.g., `data_raw/some_subdir/data_coinbase_all_1h_part1`).

### 3. Run the Complete Pipeline

```bash
python crypto_regime_pipeline.py --data_dir ./data_raw --output_dir ./data_processed --run_all
```

The pipeline will execute all 12 steps sequentially:

1. Data ingestion & merging (Coinbase parts)
2. Coin selection (top 50 by data availability & activity)
3. Market data preprocessing (BTCD + FGI)
4. Master dataset construction (per coin)
5. Feature engineering (technical indicators)
6. **HMM regime detection** (trained **only** on training period – no look‑ahead)
7. Target construction (future regime labels)
8. Model training (Random Forest & XGBoost)
9. Ablation study (without HMM regime features)
10. Model comparison
11. SHAP analysis
12. Publication‑quality visualisation

---

## ⚙️ Running Individual Steps

You can run any specific step or a combination of steps:

```bash
# Run only HMM regime detection
python crypto_regime_pipeline.py --step hmm_regime

# Run feature engineering, then model training
python crypto_regime_pipeline.py --step feature_engineering --step train_models

# List all available steps
python crypto_regime_pipeline.py --list_steps
```

---

## 📊 Key Features & Fixes

- **No Look‑Ahead Bias**  
  - HMM is **fitted only on the training period** (first 80% of sorted data).  
  - `StandardScaler` is also fitted **only on training data**.  
  - All rolling features (`pct_change`, `rolling`, `shift`) use only past information.

- **Time‑Based Train/Test Split**  
  - All model evaluations respect temporal ordering – no shuffling.

- **Multi‑Horizon Prediction**  
  - Targets: 1h, 6h, 24h, 7d, 30d ahead.

- **Ablation Study**  
  - Compares models with vs. without HMM regime features to quantify their contribution.

- **SHAP Explanations**  
  - Provides feature importance rankings and summary plots for the best models.

- **Publication‑Ready Outputs**  
  - All figures and tables are saved in `publication_figures/` and `publication_tables/`.

---

## 🧠 HMM Regime Detection

The Hidden Markov Model is trained on **training‑period only** and uses the following features:
- `return_24h`
- `volatility_24h`
- `rsi_14`
- `btcd`
- `fgi_value`

Regimes are then predicted for the entire dataset using the same model, ensuring that no future data influences the regime labels used as features for the downstream classifiers.

---

## 📈 Outputs

After a successful run, you will find:

- **Model metrics** (accuracy, precision, recall, F1) for RF and XGB with and without HMM.
- **Feature importance** tables.
- **SHAP summary plots** (`*_summary.png`) and bar plots (`*_bar.png`).
- **Comparison tables** of all model variants.
- **Regime visualisations** (distribution, timeline, transition matrix, regime characteristics).
- **Performance comparison figures** (accuracy, F1, ablation impact).

All outputs are organised in the specified `--output_dir`.

---

## 🛠 Configuration

You can adjust hyperparameters by editing the `Config` class inside `crypto_regime_pipeline.py`. Key parameters include:

- `TOP_N = 50` – number of coins to select.
- `TRAIN_RATIO = 0.8` – train/test split ratio.
- `HMM_N_COMPONENTS = 4` – number of hidden regimes.
- `RF_N_ESTIMATORS = 300`, `XGB_N_ESTIMATORS = 300` – tree counts.
- `HORIZONS` – prediction horizons in hours.

---

## 📜 License

This project is provided for research and educational purposes. Use at your own risk.

---

## 🙏 Acknowledgments

Built with:
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [hmmlearn](https://hmmlearn.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)

---

## ✉️ Contact

For issues or questions, please open an issue in the repository or reach out to the maintainers.

---

**Happy forecasting!** 🚀