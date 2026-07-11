import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Config:

    DATA_DIR: Path = Path("../data_raw")
    OUTPUT_DIR: Path = Path("../data_processed")

    COINBASE_PART1: Path = None
    COINBASE_PART2: Path = None

    BTCD_BTC_FILES: List[Path] = []
    BTCD_GLOBAL_FILE: Path = None

    MIN_ROWS: int = 30000
    ACTIVE_DATE: str = "2026-04-01"
    TOP_N: int = 50
    STABLECOINS: List[str] = [
        "USDT-USD", "USDC-USD", "DAI-USD", "PYUSD-USD", "FDUSD-USD",
        "TUSD-USD", "USDP-USD", "GUSD-USD", "EURC-USD"
    ]

    RSI_PERIOD: int = 14
    ATR_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    VOLATILITY_WINDOWS: List[int] = [24, 72]
    RETURN_WINDOWS: List[int] = [1, 6, 24]
    VOLUME_MA_WINDOW: int = 24

    HMM_N_COMPONENTS: int = 4
    HMM_COVARIANCE_TYPE: str = "full"
    HMM_N_ITER: int = 500
    HMM_RANDOM_STATE: int = 42
    HMM_FEATURES: List[str] = [
        "return_24h", "volatility_24h", "rsi_14", "btcd", "fgi_value"
    ]
    REGIME_LABELS: Dict[int, str] = {
        0: "Bull_Breakout",
        1: "Fear_Market",
        2: "Stable_Bull",
        3: "Panic_Market"
    }

    HORIZONS: Dict[str, int] = {
        "1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720
    }

    TRAIN_RATIO: float = 0.8

    RF_N_ESTIMATORS: int = 300
    RF_MAX_DEPTH: int = 15
    RF_MIN_SAMPLES_LEAF: int = 5
    RF_RANDOM_STATE: int = 42

    XGB_N_ESTIMATORS: int = 300
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.05
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_RANDOM_STATE: int = 42

    @staticmethod
    def get_xgb_tree_method() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                print("[XGB] GPU detected via PyTorch -> using tree_method='gpu_hist'")
                return "gpu_hist"
        except ImportError:
            pass
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                print("[XGB] GPU detected via nvidia-smi -> using tree_method='gpu_hist'")
                return "gpu_hist"
        except Exception:
            pass
        print("[XGB] No GPU detected -> using tree_method='hist' (CPU)")
        return "hist"

    FEATURES_NO_HMM: List[str] = [
        "open", "high", "low", "close", "volume",
        "btcd", "fgi_value",
        "return_1h", "return_6h", "return_24h",
        "volatility_24h", "volatility_72h",
        "rsi_14", "roc_24",
        "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "bb_width", "atr_14",
        "volume_change", "volume_ma24_ratio",
        "btcd_change_24h", "fgi_change_24h"
    ]

    TARGETS: List[str] = ["target_1h", "target_6h", "target_24h", "target_7d", "target_30d"]

    DROP_COLS: List[str] = ["timestamp", "symbol", "coin", "fgi_class", "market_regime_label"]

    def __init__(self, data_dir: str = "../data_raw", output_dir: str = "../data_processed"):
        self.DATA_DIR = Path(data_dir)
        self.OUTPUT_DIR = Path(output_dir)

        self.COINBASE_PART1, self.COINBASE_PART2 = self._detect_coinbase_dirs()
        self.BTCD_BTC_FILES = [
            self.DATA_DIR / "data_btcd/Bitcoin_7_9_2022-9_8_2022_historical_data_coinmarketcap.csv",
            self.DATA_DIR / "data_btcd/Bitcoin_11_1_2025-11_20_2025_historical_data_coinmarketcap.csv",
            self.DATA_DIR / "data_btcd/Bitcoin_11_3_2025-5_2_2026_historical_data_coinmarketcap.csv"
        ]
        self.BTCD_GLOBAL_FILE = self.DATA_DIR / "data_btcd/CoinGecko-GlobalCryptoMktCap-2026-05-02 (2).csv"

        for subdir in [
            "coinbase_merged", "coin_selection", "btcd", "fgi",
            "master_data", "feature_data", "regime_data",
            "model_data", "model_results_rf", "model_results_xgb",
            "ablation_results", "model_comparison",
            "shap_results", "publication_figures", "publication_tables"
        ]:
            (self.OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    def _detect_coinbase_dirs(self) -> Tuple[Path, Path]:
        candidates_part1 = []
        candidates_part2 = []

        for d in ["data_coinbase_all_1h_part1", "data_coinbase_all_1h_part2"]:
            p = self.DATA_DIR / d
            if p.is_dir():
                if "part1" in d.lower():
                    candidates_part1.append(p)
                else:
                    candidates_part2.append(p)

        if self.DATA_DIR.exists():
            for sub in self.DATA_DIR.iterdir():
                if sub.is_dir():
                    for d in ["data_coinbase_all_1h_part1", "data_coinbase_all_1h_part2"]:
                        p = sub / d
                        if p.is_dir():
                            if "part1" in d.lower():
                                candidates_part1.append(p)
                            else:
                                candidates_part2.append(p)

        if not candidates_part1 and not candidates_part2 and self.DATA_DIR.exists():
            for sub in self.DATA_DIR.iterdir():
                if sub.is_dir() and any(sub.glob("*.csv")):
                    sample = list(sub.glob("*.csv"))[:1]
                    if sample:
                        try:
                            df = pd.read_csv(sample[0], nrows=5)
                            if "time" in df.columns or "timestamp" in df.columns:
                                if not candidates_part1:
                                    candidates_part1.append(sub)
                                elif not candidates_part2:
                                    candidates_part2.append(sub)
                        except Exception:
                            pass

        part1 = candidates_part1[0] if candidates_part1 else self.DATA_DIR / "data_coinbase_all_1h_part1"
        part2 = candidates_part2[0] if candidates_part2 else self.DATA_DIR / "data_coinbase_all_1h_part2"

        if candidates_part1:
            print(f"[Config] Coinbase Part1 auto-detected: {part1}")
        else:
            print(f"[Config] Coinbase Part1 NOT found (default: {part1})")
        if candidates_part2:
            print(f"[Config] Coinbase Part2 auto-detected: {part2}")
        else:
            print(f"[Config] Coinbase Part2 NOT found (default: {part2})")

        return part1, part2


class DataIngestion:

    def __init__(self, config: Config):
        self.config = config

    def merge_coinbase_parts(self) -> pd.DataFrame:
        cfg = self.config
        output_dir = cfg.OUTPUT_DIR / "coinbase_merged"
        output_dir.mkdir(parents=True, exist_ok=True)

        files_part1 = {f.name for f in cfg.COINBASE_PART1.glob("*.csv")} if cfg.COINBASE_PART1.exists() else set()
        files_part2 = {f.name for f in cfg.COINBASE_PART2.glob("*.csv")} if cfg.COINBASE_PART2.exists() else set()
        all_files = sorted(files_part1.union(files_part2))

        print(f"[Step 1] Coinbase: Part1={len(files_part1)}, Part2={len(files_part2)}, Total unique={len(all_files)}")

        summary = []
        for coin in tqdm(all_files, desc="Merging Coinbase"):
            dfs = []
            for part_dir in [cfg.COINBASE_PART1, cfg.COINBASE_PART2]:
                filepath = part_dir / coin
                if filepath.exists():
                    dfs.append(self._load_single_coin(filepath))

            if not dfs:
                continue

            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            df.to_csv(output_dir / coin, index=False)

            summary.append({
                "coin": coin,
                "rows": len(df),
                "start_date": df["timestamp"].min(),
                "end_date": df["timestamp"].max()
            })

        if not summary:
            print("[Step 1] WARNING: No CSV files found. Checking directory structure...")
            print(f"  DATA_DIR: {cfg.DATA_DIR} (exists: {cfg.DATA_DIR.exists()})")
            print(f"  COINBASE_PART1: {cfg.COINBASE_PART1} (exists: {cfg.COINBASE_PART1.exists()})")
            print(f"  COINBASE_PART2: {cfg.COINBASE_PART2} (exists: {cfg.COINBASE_PART2.exists()})")
            if cfg.DATA_DIR.exists():
                subdirs = [d.name for d in cfg.DATA_DIR.iterdir() if d.is_dir()]
                print(f"  Subdirectories in DATA_DIR: {subdirs}")
                csv_dirs = [d.name for d in cfg.DATA_DIR.iterdir()
                            if d.is_dir() and any(d.glob("*.csv"))]
                print(f"  Dirs containing CSVs: {csv_dirs}")
                if csv_dirs:
                    print("\n[Step 1] Hint: Set --data_dir to the parent folder that contains")
                    print("  'data_coinbase_all_1h_part1' and 'data_coinbase_all_1h_part2' directly.")
                    print(f"  Or if your CSVs are directly in: {csv_dirs[0]}")
                    print(f"  Try: --data_dir {cfg.DATA_DIR}")
            raise FileNotFoundError(
                "No Coinbase CSV files found. Please check --data_dir path. "
                f"Expected subdirectories: data_coinbase_all_1h_part1, data_coinbase_all_1h_part2"
            )

        summary_df = pd.DataFrame(summary).sort_values("rows", ascending=False)
        summary_df.to_csv(cfg.OUTPUT_DIR / "coin_merge_summary.csv", index=False)
        print(f"[Step 1] Merged {len(summary_df)} coins. Saved to coin_merge_summary.csv")
        return summary_df

    @staticmethod
    def _load_single_coin(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class CoinSelection:

    def __init__(self, config: Config):
        self.config = config

    def select_coins(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        output_dir = cfg.OUTPUT_DIR / "coin_selection"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = summary_df.copy()
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["symbol"] = df["coin"].str.replace(".csv", "", regex=False)

        print(f"[Step 2] Total raw coins: {len(df)}")

        df = df[df["rows"] >= cfg.MIN_ROWS].copy()
        print(f"[Step 2] After min_rows filter (>= {cfg.MIN_ROWS}): {len(df)}")

        active_date = pd.Timestamp(cfg.ACTIVE_DATE)
        df = df[df["end_date"] >= active_date].copy()
        print(f"[Step 2] After active date filter (>= {cfg.ACTIVE_DATE}): {len(df)}")

        df = df[~df["symbol"].isin(cfg.STABLECOINS)].copy()
        print(f"[Step 2] After stablecoin removal: {len(df)}")

        df = df.sort_values(by=["rows", "start_date"], ascending=[False, True])
        selected = df.head(cfg.TOP_N).copy().reset_index(drop=True)
        print(f"[Step 2] Selected top {cfg.TOP_N} coins.")

        selected.to_csv(output_dir / "selected_coins_top50.csv", index=False)
        selected[["coin"]].to_csv(output_dir / "selected_coins.csv", index=False)

        summary = {
            "total_coin_raw": len(summary_df),
            "after_min_rows": len(df) + (cfg.TOP_N - len(selected)) if len(df) > cfg.TOP_N else len(df),
            "final_selected": len(selected),
            "min_rows_selected": int(selected["rows"].min()),
            "max_rows_selected": int(selected["rows"].max()),
            "avg_rows_selected": round(float(selected["rows"].mean()), 2)
        }
        pd.DataFrame([summary]).to_csv(output_dir / "coin_selection_summary.csv", index=False)

        return selected


class MarketDataPreprocessor:

    def __init__(self, config: Config):
        self.config = config

    def process_btcd(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        btcd_dir = cfg.OUTPUT_DIR / "btcd"
        btcd_dir.mkdir(parents=True, exist_ok=True)

        btc_frames = []
        for filepath in cfg.BTCD_BTC_FILES:
            if filepath.exists():
                df = pd.read_csv(filepath, sep=";")
                df["timeOpen"] = pd.to_datetime(df["timeOpen"], utc=True)
                btc_frames.append(pd.DataFrame({
                    "timestamp": df["timeOpen"],
                    "btc_marketcap": pd.to_numeric(df["marketCap"], errors="coerce")
                }))

        btc_df = pd.concat(btc_frames, ignore_index=True)
        btc_df = btc_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        btc_df["timestamp"] = btc_df["timestamp"].dt.tz_localize(None)
        btc_df["btc_marketcap"] = btc_df["btc_marketcap"].astype("float64")

        global_df = pd.read_csv(cfg.BTCD_GLOBAL_FILE)
        global_df["timestamp"] = pd.to_datetime(global_df["snapped_at"], unit="ms", utc=True)
        global_df = global_df.rename(columns={"market_cap": "total_marketcap"})
        global_df = global_df[["timestamp", "total_marketcap"]]
        global_df["timestamp"] = global_df["timestamp"].dt.tz_localize(None)
        global_df["total_marketcap"] = global_df["total_marketcap"].astype("float64")

        btcd_df = pd.merge(btc_df, global_df, on="timestamp", how="inner")
        btcd_df["btcd"] = (btcd_df["btc_marketcap"] / btcd_df["total_marketcap"]) * 100
        btcd_df = btcd_df[["timestamp", "btcd"]].dropna().sort_values("timestamp").reset_index(drop=True)

        btcd_df.to_csv(btcd_dir / "btcd_daily.csv", index=False)

        btcd_hourly = btcd_df.set_index("timestamp").resample("1h").ffill().reset_index()
        btcd_hourly.to_csv(btcd_dir / "btcd_hourly.csv", index=False)

        print(f"[Step 3] BTCD: {len(btcd_hourly)} hourly rows, "
              f"range [{btcd_hourly['timestamp'].min()}] to [{btcd_hourly['timestamp'].max()}]")
        return btcd_df, btcd_hourly

    def process_fgi(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        fgi_dir = cfg.OUTPUT_DIR / "fgi"
        fgi_dir.mkdir(parents=True, exist_ok=True)

        import requests

        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        fgi = pd.DataFrame(response.json()["data"])
        fgi["timestamp"] = pd.to_datetime(
            pd.to_numeric(fgi["timestamp"], errors="coerce"), unit="s"
        )
        fgi["fgi_value"] = pd.to_numeric(fgi["value"], errors="coerce")
        fgi = fgi.rename(columns={"value_classification": "fgi_class"})
        fgi = fgi[["timestamp", "fgi_value", "fgi_class"]]
        fgi = fgi.sort_values("timestamp").reset_index(drop=True)

        fgi = fgi[fgi["timestamp"] >= "2022-01-01"].copy()

        fgi = fgi.set_index("timestamp")
        full_range = pd.date_range(start=fgi.index.min(), end=fgi.index.max(), freq="D")
        fgi = fgi.reindex(full_range).ffill().reset_index().rename(columns={"index": "timestamp"})

        fgi.to_csv(fgi_dir / "fgi_daily.csv", index=False)

        fgi_hourly = fgi.set_index("timestamp").resample("1h").ffill().reset_index()
        fgi_hourly.to_csv(fgi_dir / "fgi_hourly.csv", index=False)

        print(f"[Step 3] FGI: {len(fgi_hourly)} hourly rows, "
              f"range [{fgi_hourly['timestamp'].min()}] to [{fgi_hourly['timestamp'].max()}]")
        return fgi, fgi_hourly


class MasterDatasetBuilder:

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> int:
        cfg = self.config
        output_dir = cfg.OUTPUT_DIR / "master_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        btcd = pd.read_csv(cfg.OUTPUT_DIR / "btcd/btcd_hourly.csv", parse_dates=["timestamp"])
        fgi = pd.read_csv(cfg.OUTPUT_DIR / "fgi/fgi_hourly.csv", parse_dates=["timestamp"])
        selected = pd.read_csv(cfg.OUTPUT_DIR / "coin_selection/selected_coins_top50.csv")

        coin_list = selected["coin"].tolist()
        print(f"[Step 4] Building master dataset for {len(coin_list)} coins...")

        failed = []
        success_count = 0

        for coin in tqdm(coin_list, desc="Building master"):
            try:
                paths = []
                for part_dir in [cfg.COINBASE_PART1, cfg.COINBASE_PART2]:
                    filepath = part_dir / coin
                    if filepath.exists():
                        paths.append(filepath)

                if not paths:
                    failed.append((coin, "File not found"))
                    continue

                dfs = []
                for p in paths:
                    temp = pd.read_csv(p)
                    temp["timestamp"] = pd.to_datetime(temp["time"], unit="s")
                    dfs.append(temp)

                coin_df = pd.concat(dfs, ignore_index=True)
                coin_df = coin_df.sort_values("timestamp").drop_duplicates(
                    subset=["timestamp"], keep="last"
                ).reset_index(drop=True)

                cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
                        if c in coin_df.columns]
                coin_df = coin_df[cols]

                coin_df = pd.merge(coin_df, btcd, on="timestamp", how="inner")
                coin_df = pd.merge(coin_df, fgi, on="timestamp", how="inner")
                coin_df = coin_df.sort_values("timestamp").reset_index(drop=True)

                coin_df.to_csv(output_dir / coin, index=False)
                success_count += 1

            except Exception as e:
                failed.append((coin, str(e)))

        if failed:
            print(f"[Step 4] Failed: {len(failed)} coins")
            for name, err in failed[:5]:
                print(f"  - {name}: {err}")

        print(f"[Step 4] Successfully built {success_count}/{len(coin_list)} master datasets.")
        return success_count


class FeatureEngineer:

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return ema_fast, ema_slow, macd, macd_signal, macd_hist

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.copy().sort_values("timestamp")

        for w in cfg.RETURN_WINDOWS:
            df[f"return_{w}h"] = df["close"].pct_change(w)

        for w in cfg.VOLATILITY_WINDOWS:
            df[f"volatility_{w}h"] = df["return_1h"].rolling(w).std()

        df["rsi_14"] = self.compute_rsi(df["close"], cfg.RSI_PERIOD)

        df["roc_24"] = df["close"] / df["close"].shift(24) - 1

        df["ema_12"], df["ema_26"], df["macd"], df["macd_signal"], df["macd_hist"] = \
            self.compute_macd(df["close"], cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)

        sma = df["close"].rolling(cfg.BOLLINGER_PERIOD).mean()
        std = df["close"].rolling(cfg.BOLLINGER_PERIOD).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        df["bb_width"] = (upper - lower) / sma

        df["atr_14"] = self.compute_atr(df, cfg.ATR_PERIOD)

        df["volume_change"] = df["volume"].pct_change()
        volume_ma = df["volume"].rolling(cfg.VOLUME_MA_WINDOW).mean()
        df["volume_ma24_ratio"] = df["volume"] / volume_ma

        df["btcd_change_24h"] = df["btcd"].pct_change(24)
        df["fgi_change_24h"] = df["fgi_value"].pct_change(24)

        return df

    def process_all_coins(self) -> int:
        cfg = self.config
        master_dir = cfg.OUTPUT_DIR / "master_data"
        output_dir = cfg.OUTPUT_DIR / "feature_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(master_dir.glob("*.csv"))
        print(f"[Step 5] Processing {len(files)} coin files for feature engineering...")

        for file in tqdm(files, desc="Feature engineering"):
            df = pd.read_csv(file, parse_dates=["timestamp"])
            df = self.create_features(df)
            df = df.dropna().reset_index(drop=True)

            out_name = file.stem + "_features.csv"
            df.to_csv(output_dir / out_name, index=False)

        print(f"[Step 5] Feature engineering complete. Output in {output_dir}")
        return len(list(output_dir.glob("*_features.csv")))


class HMMRegimeDetector:

    def __init__(self, config: Config):
        self.config = config
        self.hmm_model = None
        self.scaler = None

    def detect_regimes(self) -> None:
        cfg = self.config
        feature_dir = cfg.OUTPUT_DIR / "feature_data"
        regime_dir = cfg.OUTPUT_DIR / "regime_data"
        regime_dir.mkdir(parents=True, exist_ok=True)

        feature_files = sorted(feature_dir.glob("*_features.csv"))
        print(f"[Step 6] HMM regime detection for {len(feature_files)} coins...")

        all_dfs = []
        for f in feature_files:
            df = pd.read_csv(f)
            df["_source_file"] = f.name
            all_dfs.append(df)

        all_data = pd.concat(all_dfs, ignore_index=True)
        all_data["timestamp"] = pd.to_datetime(all_data["timestamp"], format="mixed")
        all_data = all_data.sort_values("timestamp").reset_index(drop=True)

        split_idx = int(len(all_data) * cfg.TRAIN_RATIO)
        train_data = all_data.iloc[:split_idx].copy()
        print(f"[Step 6] Train size: {len(train_data)}, "
              f"Test size: {len(all_data) - split_idx}")
        print(f"[Step 6] Train period: [{train_data['timestamp'].min()}] to "
              f"[{train_data['timestamp'].max()}]")

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train = train_data[cfg.HMM_FEATURES].dropna()
        self.scaler.fit(X_train)
        print(f"[Step 6] Scaler fitted on {len(X_train)} training rows only.")

        from hmmlearn.hmm import GaussianHMM
        X_train_scaled = self.scaler.transform(X_train)

        self.hmm_model = GaussianHMM(
            n_components=cfg.HMM_N_COMPONENTS,
            covariance_type=cfg.HMM_COVARIANCE_TYPE,
            n_iter=cfg.HMM_N_ITER,
            random_state=cfg.HMM_RANDOM_STATE
        )
        self.hmm_model.fit(X_train_scaled)
        print(f"[Step 6] HMM trained on training period only ({cfg.HMM_N_COMPONENTS} components).")

        results = []
        for f in tqdm(feature_files, desc="HMM prediction"):
            try:
                df = pd.read_csv(f)
                X = df[cfg.HMM_FEATURES].copy()
                X_scaled = self.scaler.transform(X)

                mask = ~np.isnan(X_scaled).any(axis=1)
                df_valid = df[mask].copy()
                X_valid_scaled = X_scaled[mask]

                states = self.hmm_model.predict(X_valid_scaled)
                df_valid["market_regime"] = states
                df_valid["market_regime_label"] = df_valid["market_regime"].map(cfg.REGIME_LABELS)

                out_name = f.name.replace("_features.csv", "_regime.csv")
                df_valid.to_csv(regime_dir / out_name, index=False)
                results.append([f.stem, len(df_valid)])

            except Exception as e:
                print(f"  [WARN] {f.name}: {e}")

        summary_df = pd.DataFrame(results, columns=["coin", "rows"])
        summary_df.to_csv(regime_dir / "regime_summary.csv", index=False)
        print(f"[Step 6] Regime detection complete. {len(results)} files saved.")

        if results:
            sample_file = regime_dir / (feature_files[0].stem.replace("_features", "_regime") + ".csv")
            if sample_file.exists():
                sample = pd.read_csv(sample_file)
                print("[Step 6] Regime distribution (first coin):")
                print(sample["market_regime"].value_counts().sort_index())


class TargetConstructor:

    def __init__(self, config: Config):
        self.config = config

    def construct_targets(self) -> None:
        cfg = self.config
        regime_dir = cfg.OUTPUT_DIR / "regime_data"
        output_dir = cfg.OUTPUT_DIR / "model_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        regime_files = sorted(regime_dir.glob("*_regime.csv"))
        print(f"[Step 7] Constructing targets for {len(regime_files)} coins...")

        summary = []
        for file in tqdm(regime_files, desc="Target construction"):
            df = pd.read_csv(file)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

            for horizon_name, horizon_step in cfg.HORIZONS.items():
                df[f"target_{horizon_name}"] = df["market_regime"].shift(-horizon_step)

            df = df.dropna(subset=[f"target_{h}" for h in cfg.HORIZONS.keys()]).reset_index(drop=True)

            for col in cfg.TARGETS:
                df[col] = df[col].astype(int)

            out_name = file.name.replace("_regime.csv", "_model.csv")
            df.to_csv(output_dir / out_name, index=False)
            summary.append([file.stem, len(df)])

        summary_df = pd.DataFrame(summary, columns=["coin", "rows"])
        summary_df.to_csv(output_dir / "model_summary.csv", index=False)
        print(f"[Step 7] Target construction complete. {len(summary)} files saved.")


class ModelTrainer:

    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.features = None

    def _load_and_prepare_dataset(self) -> None:
        cfg = self.config
        model_dir = cfg.OUTPUT_DIR / "model_data"

        files = sorted(model_dir.glob("*_model.csv"))
        print(f"[ModelTrainer] Loading {len(files)} model files...")

        all_dfs = []
        for file in files:
            df = pd.read_csv(file)
            coin_name = file.stem.replace("_model", "")
            df["coin"] = coin_name
            all_dfs.append(df)

        self.dataset = pd.concat(all_dfs, ignore_index=True)
        self.dataset["timestamp"] = pd.to_datetime(self.dataset["timestamp"], format="mixed")
        self.dataset = self.dataset.sort_values("timestamp").reset_index(drop=True)

        self.features = [
            c for c in self.dataset.columns
            if c not in cfg.TARGETS + cfg.DROP_COLS
        ]
        print(f"[ModelTrainer] Dataset: {self.dataset.shape}, Features: {len(self.features)}")

    def _time_split(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        temp = df.dropna(subset=[target])
        split_idx = int(len(temp) * self.config.TRAIN_RATIO)
        return temp.iloc[:split_idx], temp.iloc[split_idx:]

    def train_random_forest(self) -> pd.DataFrame:
        cfg = self.config
        output_dir = cfg.OUTPUT_DIR / "model_results_rf"
        output_dir.mkdir(parents=True, exist_ok=True)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from joblib import dump, load

        self._load_and_prepare_dataset()
        results = []
        importance_rows = []

        for target in cfg.TARGETS:
            model_file = output_dir / f"rf_{target}.joblib"
            if model_file.exists():
                print(f"[RF] Skipping {target} (already trained, evaluating...)")
                rf = load(model_file)
                train_df, test_df = self._time_split(self.dataset, target)
                X_test, y_test = test_df[self.features], test_df[target]
                pred = rf.predict(X_test)
                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
                results.append([target, len(train_df), len(test_df), acc, prec, rec, f1])
                for feat, imp in zip(self.features, rf.feature_importances_):
                    importance_rows.append([target, feat, imp])
                print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
                continue
            print(f"\n{'='*50}\n[RF] Training: {target}\n{'='*50}")

            train_df, test_df = self._time_split(self.dataset, target)
            X_train, y_train = train_df[self.features], train_df[target]
            X_test, y_test = test_df[self.features], test_df[target]

            rf = RandomForestClassifier(
                n_estimators=cfg.RF_N_ESTIMATORS,
                max_depth=cfg.RF_MAX_DEPTH,
                min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
                random_state=cfg.RF_RANDOM_STATE,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

            results.append([target, len(train_df), len(test_df), acc, prec, rec, f1])
            dump(rf, output_dir / f"rf_{target}.joblib")

            for feat, imp in zip(self.features, rf.feature_importances_):
                importance_rows.append([target, feat, imp])

            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")

        metrics_df = pd.DataFrame(results, columns=[
            "target", "train_rows", "test_rows", "accuracy", "precision", "recall", "f1_score"
        ])
        metrics_df.to_csv(output_dir / "rf_metrics.csv", index=False)

        importance_df = pd.DataFrame(importance_rows, columns=["target", "feature", "importance"])
        importance_df = importance_df.sort_values(["target", "importance"], ascending=[True, False])
        importance_df.to_csv(output_dir / "rf_feature_importance.csv", index=False)

        print(f"\n[RF] Results saved to {output_dir}")
        return metrics_df

    def train_xgboost(self) -> pd.DataFrame:
        cfg = self.config
        output_dir = cfg.OUTPUT_DIR / "model_results_xgb"
        output_dir.mkdir(parents=True, exist_ok=True)

        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from joblib import dump, load

        self._load_and_prepare_dataset()
        results = []
        importance_rows = []

        for target in cfg.TARGETS:
            model_file = output_dir / f"xgb_{target}.joblib"
            if model_file.exists():
                print(f"[XGB] Skipping {target} (already trained, evaluating...)")
                model = load(model_file)
                train_df, test_df = self._time_split(self.dataset, target)
                X_test, y_test = test_df[self.features], test_df[target]
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
                results.append([target, len(train_df), len(test_df), acc, prec, rec, f1])
                for feat, imp in zip(self.features, model.feature_importances_):
                    importance_rows.append([target, feat, imp])
                print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
                continue
            print(f"\n{'='*60}\n[XGB] Training: {target}\n{'='*60}")

            train_df, test_df = self._time_split(self.dataset, target)
            X_train, y_train = train_df[self.features], train_df[target]
            X_test, y_test = test_df[self.features], test_df[target]

            n_classes = y_train.nunique()

            model = XGBClassifier(
                n_estimators=cfg.XGB_N_ESTIMATORS,
                max_depth=cfg.XGB_MAX_DEPTH,
                learning_rate=cfg.XGB_LEARNING_RATE,
                subsample=cfg.XGB_SUBSAMPLE,
                colsample_bytree=cfg.XGB_COLSAMPLE_BYTREE,
                objective="multi:softmax",
                num_class=n_classes,
                tree_method=cfg.get_xgb_tree_method(),
                random_state=cfg.XGB_RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss"
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

            results.append([target, len(train_df), len(test_df), acc, prec, rec, f1])
            dump(model, output_dir / f"xgb_{target}.joblib")

            for feat, imp in zip(self.features, model.feature_importances_):
                importance_rows.append([target, feat, imp])

            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")

        metrics_df = pd.DataFrame(results, columns=[
            "target", "train_rows", "test_rows", "accuracy", "precision", "recall", "f1_score"
        ])
        metrics_df.to_csv(output_dir / "xgb_metrics.csv", index=False)

        importance_df = pd.DataFrame(importance_rows, columns=["target", "feature", "importance"])
        importance_df = importance_df.sort_values(["target", "importance"], ascending=[True, False])
        importance_df.to_csv(output_dir / "xgb_feature_importance.csv", index=False)

        print(f"\n[XGB] Results saved to {output_dir}")
        return metrics_df


class AblationStudy:

    def __init__(self, config: Config):
        self.config = config

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        output_dir = cfg.OUTPUT_DIR / "ablation_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = cfg.OUTPUT_DIR / "model_data"
        files = sorted(model_dir.glob("*_model.csv"))
        features = cfg.FEATURES_NO_HMM

        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        rf_csv = output_dir / "rf_without_hmm.csv"
        xgb_csv = output_dir / "xgb_without_hmm.csv"

        if rf_csv.exists() and xgb_csv.exists():
            existing_rf = pd.read_csv(rf_csv)
            existing_xgb = pd.read_csv(xgb_csv)
            done_targets = set(existing_rf["target"].tolist()) & set(existing_xgb["target"].tolist())
            remaining = [t for t in cfg.TARGETS if t not in done_targets]
            if not remaining:
                print(f"[Ablation] All targets already completed. Skipping.")
                return existing_rf, existing_xgb
            print(f"[Ablation] Resuming: skipping {len(done_targets)} done, {len(remaining)} remaining")
            rf_results = existing_rf.to_dict("records")
            xgb_results = existing_xgb.to_dict("records")
        else:
            rf_results = []
            xgb_results = []
            remaining = cfg.TARGETS

        for target in remaining:
            print(f"\n{'='*60}\n[Ablation] Target: {target}\n{'='*60}")

            datasets = []
            for file in files:
                df = pd.read_csv(file)
                cols = ["timestamp"] + features + [target]
                available_cols = [c for c in cols if c in df.columns]
                datasets.append(df[available_cols])

            dataset = pd.concat(datasets, ignore_index=True)
            dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], format="mixed")
            dataset = dataset.sort_values("timestamp").reset_index(drop=True)
            dataset = dataset.dropna(subset=[target])

            split_idx = int(len(dataset) * cfg.TRAIN_RATIO)
            train_df = dataset.iloc[:split_idx]
            test_df = dataset.iloc[split_idx:]

            X_train = train_df[features]
            y_train = train_df[target]
            X_test = test_df[features]
            y_test = test_df[target]

            n_classes = y_train.nunique()

            rf = RandomForestClassifier(
                n_estimators=cfg.RF_N_ESTIMATORS,
                max_depth=cfg.RF_MAX_DEPTH,
                min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
                random_state=cfg.RF_RANDOM_STATE,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)

            rf_results.append({
                "target": target,
                "accuracy": accuracy_score(y_test, pred_rf),
                "precision": precision_score(y_test, pred_rf, average="weighted", zero_division=0),
                "recall": recall_score(y_test, pred_rf, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, pred_rf, average="weighted", zero_division=0)
            })

            xgb = XGBClassifier(
                n_estimators=cfg.XGB_N_ESTIMATORS,
                max_depth=cfg.XGB_MAX_DEPTH,
                learning_rate=cfg.XGB_LEARNING_RATE,
                subsample=cfg.XGB_SUBSAMPLE,
                colsample_bytree=cfg.XGB_COLSAMPLE_BYTREE,
                objective="multi:softmax",
                num_class=n_classes,
                tree_method=cfg.get_xgb_tree_method(),
                random_state=cfg.XGB_RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss"
            )
            xgb.fit(X_train, y_train)
            pred_xgb = xgb.predict(X_test)

            xgb_results.append({
                "target": target,
                "accuracy": accuracy_score(y_test, pred_xgb),
                "precision": precision_score(y_test, pred_xgb, average="weighted", zero_division=0),
                "recall": recall_score(y_test, pred_xgb, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, pred_xgb, average="weighted", zero_division=0)
            })

            print(f"  RF F1: {rf_results[-1]['f1_score']:.4f}, XGB F1: {xgb_results[-1]['f1_score']:.4f}")

            pd.DataFrame(rf_results).to_csv(rf_csv, index=False)
            pd.DataFrame(xgb_results).to_csv(xgb_csv, index=False)

        rf_df = pd.DataFrame(rf_results, columns=["target", "accuracy", "precision", "recall", "f1_score"])
        xgb_df = pd.DataFrame(xgb_results, columns=["target", "accuracy", "precision", "recall", "f1_score"])

        rf_df.to_csv(output_dir / "rf_without_hmm.csv", index=False)
        xgb_df.to_csv(output_dir / "xgb_without_hmm.csv", index=False)

        print(f"\n[Ablation] Results saved to {output_dir}")
        return rf_df, xgb_df


class ModelComparison:

    def __init__(self, config: Config):
        self.config = config

    def compare(self) -> pd.DataFrame:
        cfg = self.config
        base_dir = cfg.OUTPUT_DIR
        output_dir = base_dir / "model_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        rf_hmm = pd.read_csv(base_dir / "model_results_rf/rf_metrics.csv")
        xgb_hmm = pd.read_csv(base_dir / "model_results_xgb/xgb_metrics.csv")
        rf_no_hmm = pd.read_csv(base_dir / "ablation_results/rf_without_hmm.csv")
        xgb_no_hmm = pd.read_csv(base_dir / "ablation_results/xgb_without_hmm.csv")

        rf_hmm["model"] = "RF + HMM"
        xgb_hmm["model"] = "XGB + HMM"
        rf_no_hmm["model"] = "RF"
        xgb_no_hmm["model"] = "XGB"

        comparison_df = pd.concat(
            [rf_no_hmm, rf_hmm, xgb_no_hmm, xgb_hmm],
            ignore_index=True
        )

        cols = ["model", "target", "accuracy", "precision", "recall", "f1_score"]
        available_cols = [c for c in cols if c in comparison_df.columns]
        comparison_df = comparison_df[available_cols]

        comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)

        best_models = (
            comparison_df.sort_values("accuracy", ascending=False)
            .groupby("target").head(1).reset_index(drop=True)
        )
        best_models.to_csv(output_dir / "best_models.csv", index=False)

        improvement = []
        for target in comparison_df["target"].unique():
            for base_model, hmm_model in [("RF", "RF + HMM"), ("XGB", "XGB + HMM")]:
                base_acc = comparison_df[
                    (comparison_df["model"] == base_model) &
                    (comparison_df["target"] == target)
                ]["accuracy"].values
                hmm_acc = comparison_df[
                    (comparison_df["model"] == hmm_model) &
                    (comparison_df["target"] == target)
                ]["accuracy"].values
                if len(base_acc) > 0 and len(hmm_acc) > 0 and base_acc[0] > 0:
                    gain = ((hmm_acc[0] - base_acc[0]) / base_acc[0]) * 100
                    improvement.append([target, base_model, hmm_model, gain])

        if improvement:
            imp_df = pd.DataFrame(improvement, columns=[
                "target", "base_model", "hmm_model", "improvement_percent"
            ])
            imp_df.to_csv(output_dir / "hmm_improvement.csv", index=False)
            print("\n[Comparison] HMM Improvement:")
            print(imp_df.to_string(index=False))

        print(f"\n[Comparison] Results saved to {output_dir}")
        print("\n[Comparison] Best Models per Target:")
        print(best_models.to_string(index=False))

        return comparison_df


class SHAPAnalyzer:

    def __init__(self, config: Config):
        self.config = config

    def analyze(self) -> None:
        cfg = self.config
        base_dir = cfg.OUTPUT_DIR
        output_dir = base_dir / "shap_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        import shap
        from joblib import load

        model_dir = base_dir / "model_data"
        files = sorted(model_dir.glob("*_model.csv"))
        all_dfs = [pd.read_csv(f) for f in files]
        dataset = pd.concat(all_dfs, ignore_index=True)

        features = [c for c in dataset.columns if c not in cfg.TARGETS + cfg.DROP_COLS]

        best_models = {
            "target_24h": ("RF", "rf_target_24h.joblib"),
            "target_7d": ("XGB", "xgb_target_7d.joblib"),
            "target_30d": ("XGB", "xgb_target_30d.joblib"),
        }

        for target, (model_type, model_file) in best_models.items():
            print(f"\n{'='*60}\n[SHAP] {target} ({model_type})\n{'='*60}")

            if model_type == "RF":
                model = load(base_dir / "model_results_rf" / model_file)
            else:
                model = load(base_dir / "model_results_xgb" / model_file)

            temp = dataset.dropna(subset=[target])
            X = temp[features]

            sample_size = min(3000, len(X))
            sample = X.sample(n=sample_size, random_state=42)
            print(f"  Sample size: {sample.shape}")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            print("  SHAP values computed.")

            if isinstance(shap_values, list):
                shap_for_plot = np.mean(np.abs(shap_values), axis=0)
            elif len(np.array(shap_values).shape) == 3:
                shap_for_plot = np.mean(np.abs(shap_values), axis=2)
            else:
                shap_for_plot = shap_values

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure()
            shap.summary_plot(shap_values, sample, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{target}_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{target}_bar.png", dpi=300, bbox_inches="tight")
            plt.close()

            mean_shap = np.abs(shap_for_plot).mean(axis=0)
            ranking = pd.DataFrame({
                "feature": features,
                "mean_shap": mean_shap
            }).sort_values("mean_shap", ascending=False)
            ranking.to_csv(output_dir / f"{target}_ranking.csv", index=False)

            print(f"  Top 5 features: {ranking['feature'].head(5).tolist()}")

        print(f"\n[SHAP] Results saved to {output_dir}")


class PublicationVisualizer:

    def __init__(self, config: Config):
        self.config = config

    def generate_all(self) -> None:
        cfg = self.config
        base_dir = cfg.OUTPUT_DIR
        fig_dir = base_dir / "publication_figures"
        tbl_dir = base_dir / "publication_tables"
        fig_dir.mkdir(exist_ok=True)
        tbl_dir.mkdir(exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="talk")

        comparison_df = pd.read_csv(base_dir / "model_comparison/comparison_table.csv")
        best_models = pd.read_csv(base_dir / "model_comparison/best_models.csv")

        try:
            rf_ablation = pd.read_csv(base_dir / "ablation_results/rf_without_hmm.csv")
            xgb_ablation = pd.read_csv(base_dir / "ablation_results/xgb_without_hmm.csv")
            rf_ablation["model"] = "RF (no HMM)"
            xgb_ablation["model"] = "XGB (no HMM)"
            ablation = pd.concat([rf_ablation, xgb_ablation], ignore_index=True)
        except FileNotFoundError:
            ablation = None
            print("[Visualization] Ablation results not found, skipping ablation plot.")

        self._save_grouped_bar(
            comparison_df, "target", "accuracy", "model",
            fig_dir / "figure_1_accuracy_comparison.png",
            "Accuracy Comparison Across Prediction Horizons", "Accuracy"
        )

        self._save_grouped_bar(
            comparison_df, "target", "f1_score", "model",
            fig_dir / "figure_2_f1_comparison.png",
            "F1 Score Comparison Across Prediction Horizons", "F1 Score"
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(data=best_models, x="target", y="f1_score", palette="viridis")
        plt.title("Best Performing Model Per Horizon")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(fig_dir / "figure_3_best_models.png", dpi=600)
        plt.close()

        if ablation is not None:
            self._save_grouped_bar(
                ablation, "target", "f1_score", "model",
                fig_dir / "figure_4_ablation.png",
                "Impact of Removing HMM Regime Features", "F1 Score"
            )

        self._save_table_png(best_models, tbl_dir / "table_1_best_models.png")
        self._save_table_png(comparison_df, tbl_dir / "table_4_comparison.png")

        regime_dir = base_dir / "regime_data"
        btc_regime_file = regime_dir / "BTC-USD_regime.csv"

        if btc_regime_file.exists():
            btc = pd.read_csv(btc_regime_file)
            btc["timestamp"] = pd.to_datetime(btc["timestamp"], format="mixed")

            plt.figure(figsize=(10, 6))
            sns.countplot(data=btc, x="market_regime", palette="viridis")
            plt.title("Distribution of Hidden Market Regimes (BTC)")
            plt.xlabel("Regime")
            plt.ylabel("Frequency")
            self._annotate_bars(plt)
            plt.tight_layout()
            plt.savefig(fig_dir / "figure_5_regime_distribution.png", dpi=600)
            plt.close()

            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(btc["timestamp"], btc["close"], color="black", linewidth=1.5, label="BTC Price")
            for regime in sorted(btc["market_regime"].dropna().unique()):
                mask = btc["market_regime"] == regime
                ax.fill_between(
                    btc["timestamp"], btc["close"].min(), btc["close"].max(),
                    where=mask, alpha=0.15, label=f"Regime {int(regime)}"
                )
            ax.set_title("BTC Price with Hidden Market Regimes")
            ax.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "figure_6_regime_timeline.png", dpi=600)
            plt.close()

            transition = pd.crosstab(
                btc["market_regime"], btc["market_regime"].shift(-1), normalize="index"
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(transition, annot=True, cmap="Blues", fmt=".2f")
            plt.title("HMM Regime Transition Probability Matrix")
            plt.xlabel("Next Regime")
            plt.ylabel("Current Regime")
            plt.tight_layout()
            plt.savefig(fig_dir / "figure_7_transition_matrix.png", dpi=600)
            plt.close()

            if "return_24h" in btc.columns and "volatility_24h" in btc.columns:
                regime_summary = btc.groupby("market_regime").agg({
                    "return_24h": "mean", "volatility_24h": "mean"
                }).reset_index()

                plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    data=regime_summary, x="volatility_24h", y="return_24h",
                    hue="market_regime", s=300
                )
                for _, row in regime_summary.iterrows():
                    plt.text(row["volatility_24h"], row["return_24h"],
                             f"R{int(row['market_regime'])}")
                plt.title("Characteristics of Hidden Market Regimes")
                plt.xlabel("Average Volatility")
                plt.ylabel("Average Return")
                plt.tight_layout()
                plt.savefig(fig_dir / "figure_8_regime_characteristics.png", dpi=600)
                plt.close()

        shap_dir = base_dir / "shap_results"
        shap_files = sorted(shap_dir.glob("*_ranking.csv"))
        if shap_files:
            shap_ranking = pd.read_csv(shap_files[-1])
            top10 = shap_ranking.head(10)
            plt.figure(figsize=(10, 7))
            sns.barplot(data=top10, x="mean_shap", y="feature")
            plt.title("Top 10 Most Important Features (SHAP)", fontsize=16, fontweight="bold")
            plt.xlabel("Mean |SHAP Value|")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(fig_dir / "figure_9_shap_importance.png", dpi=600, bbox_inches="tight")
            plt.close()

        print(f"\n[Visualization] All figures saved to {fig_dir}")
        print(f"[Visualization] All tables saved to {tbl_dir}")

    @staticmethod
    def _save_grouped_bar(df, x, y, hue, save_path, title, ylabel):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x=x, y=y, hue=hue)
        plt.title(title, fontsize=18, fontweight="bold")
        plt.ylabel(ylabel)
        plt.ylim(0, 1)
        PublicationVisualizer._annotate_bars(plt)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    @staticmethod
    def _annotate_bars(plt_module):
        for p in plt_module.gca().patches:
            val = p.get_height()
            if not np.isnan(val):
                plt_module.gca().annotate(
                    f"{val:.3f}",
                    (p.get_x() + p.get_width() / 2., val),
                    ha="center", va="center",
                    xytext=(0, 5), textcoords="offset points", fontsize=8
                )

    @staticmethod
    def _save_table_png(df, filename):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.8), max(2, len(df) * 0.5)))
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()


class Pipeline:

    STEPS = {
        "data_ingestion": ("Step 1: Data Ingestion (merge Coinbase)", "run_data_ingestion"),
        "coin_selection": ("Step 2: Coin Selection", "run_coin_selection"),
        "market_data": ("Step 3: Market Data (BTCD + FGI)", "run_market_data"),
        "master_dataset": ("Step 4: Master Dataset", "run_master_dataset"),
        "feature_engineering": ("Step 5: Feature Engineering", "run_feature_engineering"),
        "hmm_regime": ("Step 6: HMM Regime Detection", "run_hmm_regime"),
        "target_construction": ("Step 7: Target Construction", "run_target_construction"),
        "train_models": ("Step 8: Train Models (RF + XGB)", "run_train_models"),
        "ablation": ("Step 9: Ablation Study", "run_ablation"),
        "comparison": ("Step 10: Model Comparison", "run_comparison"),
        "shap": ("Step 11: SHAP Analysis", "run_shap"),
        "visualization": ("Step 12: Visualization", "run_visualization"),
    }

    def __init__(self, config: Config):
        self.config = config

    def run_data_ingestion(self):
        return DataIngestion(self.config).merge_coinbase_parts()

    def run_coin_selection(self, summary_df=None):
        if summary_df is None:
            summary_path = self.config.OUTPUT_DIR / "coin_merge_summary.csv"
            if not summary_path.exists():
                print("[Pipeline] coin_merge_summary.csv not found. Run data_ingestion first.")
                return None
            summary_df = pd.read_csv(summary_path)
        return CoinSelection(self.config).select_coins(summary_df)

    def run_market_data(self):
        preprocessor = MarketDataPreprocessor(self.config)
        btcd_daily, btcd_hourly = preprocessor.process_btcd()
        fgi_daily, fgi_hourly = preprocessor.process_fgi()
        return btcd_daily, fgi_daily

    def run_master_dataset(self):
        return MasterDatasetBuilder(self.config).build()

    def run_feature_engineering(self):
        return FeatureEngineer(self.config).process_all_coins()

    def run_hmm_regime(self):
        HMMRegimeDetector(self.config).detect_regimes()

    def run_target_construction(self):
        TargetConstructor(self.config).construct_targets()

    def run_train_models(self):
        trainer = ModelTrainer(self.config)
        trainer.train_random_forest()
        trainer.train_xgboost()

    def run_ablation(self):
        AblationStudy(self.config).run()

    def run_comparison(self):
        ModelComparison(self.config).compare()

    def run_shap(self):
        SHAPAnalyzer(self.config).analyze()

    def run_visualization(self):
        PublicationVisualizer(self.config).generate_all()

    def run_all(self):
        print("=" * 70)
        print("CRYPTO MARKET REGIME PREDICTION PIPELINE")
        print("=" * 70)

        step_methods = [
            ("data_ingestion", "run_data_ingestion", []),
            ("coin_selection", "run_coin_selection", []),
            ("market_data", "run_market_data", []),
            ("master_dataset", "run_master_dataset", []),
            ("feature_engineering", "run_feature_engineering", []),
            ("hmm_regime", "run_hmm_regime", []),
            ("target_construction", "run_target_construction", []),
            ("train_models", "run_train_models", []),
            ("ablation", "run_ablation", []),
            ("comparison", "run_comparison", []),
            ("shap", "run_shap", []),
            ("visualization", "run_visualization", []),
        ]

        for step_key, method_name, _ in step_methods:
            desc = self.STEPS[step_key][0]
            print(f"\n{'#'*70}")
            print(f"# {desc}")
            print(f"{'#'*70}\n")

            try:
                getattr(self, method_name)()
            except Exception as e:
                print(f"\n[ERROR] {desc} failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n[Pipeline] Stopping at {step_key}. Fix the error and re-run.")
                break

        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Market Regime Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crypto_regime_pipeline.py --run_all
  python crypto_regime_pipeline.py --step hmm_regime
  python crypto_regime_pipeline.py --step train_models --step comparison
  python crypto_regime_pipeline.py --data_dir /path/to/data --output_dir /path/to/output
        """
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data_raw",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data_processed",
        help="Path to output directory"
    )
    parser.add_argument(
        "--run_all", action="store_true",
        help="Run the complete pipeline"
    )
    parser.add_argument(
        "--step", type=str, action="append", default=[],
        choices=list(Pipeline.STEPS.keys()),
        help="Run specific step(s). Can be repeated."
    )
    parser.add_argument(
        "--list_steps", action="store_true",
        help="List all available steps"
    )

    args = parser.parse_args()

    if args.list_steps:
        print("Available steps:")
        for key, (desc, _) in Pipeline.STEPS.items():
            print(f"  {key:25s} - {desc}")
        return

    config = Config(data_dir=args.data_dir, output_dir=args.output_dir)
    pipeline = Pipeline(config)

    if args.run_all:
        pipeline.run_all()
    elif args.step:
        for step in args.step:
            desc = Pipeline.STEPS[step][0]
            method_name = Pipeline.STEPS[step][1]
            print(f"\n{'#'*70}")
            print(f"# {desc}")
            print(f"{'#'*70}\n")
            try:
                getattr(pipeline, method_name)()
            except Exception as e:
                print(f"\n[ERROR] {step} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()