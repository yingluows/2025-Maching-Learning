# select_feature.py
import os
import ast

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

TRAIN_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "train_raw_v5.csv")
TEST_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "test_raw_v5.csv")

TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v5.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v5.csv")


# ========= 皮尔逊结果配置（没有文件也能跑，会自动跳过） =========
PEARSON_DIR = os.path.join(BASE_DIR, "PEARSON_DIR")
PEARSON_FILES = [
    "Feature_prv皮尔逊分析.csv",
    "Feature_time皮尔逊分析.csv",
    "Feature_welch_1皮尔逊分析.csv",
    "Feature_welch_2皮尔逊分析.csv",
    "reference_time皮尔逊分析.csv",
    "全皮尔逊分析Feature_frequency_1.csv",
    "全皮尔逊分析Feature_frequency_2.csv",
]
PEARSON_ABS_CORR_THRESHOLD = 0.15
PEARSON_REQUIRE_SIGNIFICANT = True


def get_high_corr_feature_names(
    pearson_dir: str = PEARSON_DIR,
    files=PEARSON_FILES,
    abs_corr_thr: float = PEARSON_ABS_CORR_THRESHOLD,
    require_sig: bool = PEARSON_REQUIRE_SIGNIFICANT,
) -> set:
    selected = set()

    for fname in files:
        path = os.path.join(pearson_dir, fname)
        if not os.path.exists(path):
            print(f"⚠ 未找到皮尔逊文件: {path}，跳过")
            continue

        df = pd.read_csv(path)
        if "Feature" not in df.columns or "Correlation" not in df.columns:
            print(f"⚠ 文件 {path} 格式异常，缺少 Feature/Correlation 列，跳过")
            continue

        cond = df["Correlation"].abs() >= abs_corr_thr
        if require_sig and "Significant" in df.columns:
            cond = cond & (df["Significant"] == True)

        feats = df.loc[cond, "Feature"].astype(str).tolist()
        print(f"[Pearson] {fname} 选中 {len(feats)} 个特征 (|r|>={abs_corr_thr})")
        selected.update(feats)

    print(f"\n[Pearson] 合并后总共选中 {len(selected)} 个特征名\n")
    return selected


HIGH_CORR_FEATURES = get_high_corr_feature_names()


BRACKET_FEATURE_COLS = [
    "FDA1", "FDB1",
    *[f"frequency_lobe{i}" for i in range(11)],
    "ppg_peak", "ppg_valley",
    "rising_quarter1", "rising_quarter2", "rising_quarter3",
    "falling_quarter1", "falling_quarter2", "falling_quarter3",
    "fdppg_peak", "fdppg_peak1", "fdppg_peak2",
    "fdppg_valley", "fdppg_valley1", "fdppg_valley2",
    "sdppg_peak", "sdppg_peak1", "sdppg_peak2",
    "sdppg_valley", "sdppg_valley1",
    "sdppg_peak3", "sdppg_valley2",
    "forward_peak", "reflect_peak",
    "dicrotic_notch", "dicrotic_peak",
]


def safe_eval(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        return [None, None]
    except Exception:
        return [None, None]


def expand_bracket_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BRACKET_FEATURE_COLS:
        if col in df.columns:
            two_cols = df[col].apply(safe_eval).apply(pd.Series)
            df[f"{col}_x"] = two_cols[0]
            df[f"{col}_y"] = two_cols[1]
    return df


def deal_file(train_df: pd.DataFrame, test_df: pd.DataFrame):
    drop_not_features = [
        "data_name",
        "group_id",
        "person_day",
        "select_number",
        "total_select_number",
        "frequency_resolution",
        "cycle_number",
        "data_len",
    ]

    def _process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "label" not in df.columns:
            raise ValueError("数据中没有 label 列，请确认 deal_data.py 已经打好标签并保存。")

        df = df.drop(columns=[c for c in drop_not_features if c in df.columns], errors="ignore")

        for col in BRACKET_FEATURE_COLS:
            cx = f"{col}_x"
            cy = f"{col}_y"
            if cx in df.columns and cy in df.columns:
                df[f"{col}_mag"] = np.sqrt(df[cx] ** 2 + df[cy] ** 2)

        df = df.drop(columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns], errors="ignore")

        label = df["label"]
        user_id = df["user_id"] if "user_id" in df.columns else None

        feature_df = df.drop(columns=[c for c in ["label", "user_id"] if c in df.columns])

        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")],
            errors="ignore",
        )

        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"[deal_file] 初步处理后特征数：{feature_df.shape[1]}")

        if HIGH_CORR_FEATURES:
            keep_cols = []
            for col in feature_df.columns:
                base = col[:-4] if col.endswith("_mag") else col
                if base in HIGH_CORR_FEATURES:
                    keep_cols.append(col)

            if keep_cols:
                feature_df = feature_df[keep_cols]
                print(f"[deal_file] 皮尔逊筛选后保留特征数：{feature_df.shape[1]}")
            else:
                print("[deal_file] 皮尔逊筛选未匹配任何特征，保留全部特征。")

        if user_id is not None:
            df_processed = pd.concat(
                [user_id.reset_index(drop=True), feature_df.reset_index(drop=True), label.reset_index(drop=True)],
                axis=1,
            )
        else:
            df_processed = pd.concat([feature_df, label], axis=1)

        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)
    return train_df, test_df


def make_feature_csvs(
    train_raw_path: str = TRAIN_RAW_PATH,
    test_raw_path: str = TEST_RAW_PATH,
    train_feat_path: str = TRAIN_FEAT_PATH,
    test_feat_path: str = TEST_FEAT_PATH,
):
    train_df = pd.read_csv(train_raw_path)
    test_df = pd.read_csv(test_raw_path)

    print("原始训练集形状：", train_df.shape)
    print("原始测试集形状：", test_df.shape)

    train_df = expand_bracket_features(train_df)
    test_df = expand_bracket_features(test_df)

    train_df, test_df = deal_file(train_df, test_df)

    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    train_df.to_csv(train_feat_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_feat_path, index=False, encoding="utf-8-sig")

    print(f"✅ 处理后的训练集已保存：{train_feat_path}")
    print(f"✅ 处理后的测试集已保存：{test_feat_path}")


if __name__ == "__main__":
    make_feature_csvs()
