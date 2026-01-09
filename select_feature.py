# select_feature.py
import os
import ast
import numpy as np
import pandas as pd

# ========= 基本路径配置 =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"

DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

# deal_data 输出
TRAIN_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "train_raw_v9.csv")
TEST_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "test_raw_v9.csv")

# 本文件输出
TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v9.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v9.csv")

# ========= 皮尔逊结果配置 =========
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

# ========= 特征列配置（含 "[a,b]" 字符串列） =========
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
    """
    你的新规则：
    - BRACKET_FEATURE_COLS 相关特征（最终的 *_mag）不经过皮尔逊筛选：全部保留
    - 其它普通特征列：照旧用皮尔逊筛选
    - 删除非特征列与 user_id
    """

    drop_not_features = [
        "user_id",
        "data_name",
        "group_id",
        "person_day",
        "_date_for_limit",
        "select_number",
        "total_select_number",
        "frequency_resolution",
        "cycle_number",
        "data_len",
    ]

    def _process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "label" not in df.columns:
            raise ValueError("数据中没有 label 列，请确认 deal_data 已打好标签。")

        # 1) 删除非特征列（含 user_id）
        df = df.drop(columns=[c for c in drop_not_features if c in df.columns], errors="ignore")

        # 2) 构造 *_mag（从 *_x/_y 来）
        for col in BRACKET_FEATURE_COLS:
            cx, cy = f"{col}_x", f"{col}_y"
            if cx in df.columns and cy in df.columns:
                df[f"{col}_mag"] = np.sqrt(df[cx] ** 2 + df[cy] ** 2)

        # 2.5) 强制 mag 列为数值，避免 dtype=object 被 select_dtypes 丢弃
        mag_cols = [c for c in df.columns if c.endswith("_mag")]
        for c in mag_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 3) 删除原始 bracket 列（"[a,b]"）
        df = df.drop(columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns], errors="ignore")

        # 4) 分离 label
        y = df["label"]
        X = df.drop(columns=["label"])

        # 5) 删除 *_x, *_y（只保留 *_mag）
        X = X.drop(columns=[c for c in X.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")

        # 6) 仅保留数值特征
        X = X.select_dtypes(include=[np.number])

        # 7) 皮尔逊筛选：只筛“非 mag 的普通特征”，mag 全保留
        mag_cols = [c for c in X.columns if c.endswith("_mag")]
        normal_cols = [c for c in X.columns if not c.endswith("_mag")]

        if HIGH_CORR_FEATURES and normal_cols:
            keep_normal = [c for c in normal_cols if c in HIGH_CORR_FEATURES]
            # mag 全保留
            keep_cols = mag_cols + keep_normal
            # 去重并保持顺序
            seen = set()
            keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
            X = X[keep_cols]
            print(f"[deal_file] 保留 mag={len(mag_cols)} 个；普通特征皮尔逊保留={len(keep_normal)} 个；总={X.shape[1]}")
        else:
            # 没有皮尔逊列表或没有普通列：保留全部数值（mag 也在其中）
            print(f"[deal_file] 未执行普通特征皮尔逊筛选（HIGH_CORR_FEATURES为空或无普通列），总特征={X.shape[1]}，其中mag={len(mag_cols)}")

        out = pd.concat([X, y], axis=1)
        return out

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

    print(f"\n处理后的训练集已保存到：{train_feat_path}")
    print(f"处理后的测试集已保存到：{test_feat_path}")


if __name__ == "__main__":
    make_feature_csvs()
