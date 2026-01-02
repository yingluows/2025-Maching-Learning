# select_feature.py
"""
特征处理脚本（v7策略）
- 输入：deal_data.py 输出的 train_raw_v7.csv / test_raw_v7.csv
- 输出：train_v7.csv / test_v7.csv
策略：
1) 删除非特征列与 user_id（防止泄漏），但保留 age 供 tolerant accuracy/分析使用
2) 对 BRACKET_FEATURE_COLS 解析成 *_x/*_y，并构造 *_mag（全部构建）
3) 皮尔逊白名单筛选：只筛选非 *_mag 的特征；*_mag 无条件保留
"""

import os
import ast
import numpy as np
import pandas as pd

BASE_DIR = r"D:/2025_Stage/Code/XGB"
DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

TRAIN_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "train_raw_v7.csv")
TEST_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "test_raw_v7.csv")

TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v7.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v7.csv")

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

FORBIDDEN_COLS = ["user_id"]

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
            print(f"⚠ 文件 {path} 缺少 Feature/Correlation 列，跳过")
            continue

        cond = df["Correlation"].abs() >= abs_corr_thr
        if require_sig and "Significant" in df.columns:
            cond = cond & (df["Significant"] == True)

        feats = df.loc[cond, "Feature"].astype(str).tolist()
        print(f"[Pearson] {fname} 选中 {len(feats)} 个特征")
        selected.update(feats)

    print(f"\n[Pearson] 合并后白名单特征数：{len(selected)}\n")
    return selected


HIGH_CORR_FEATURES = get_high_corr_feature_names()


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
        # 泄漏/索引类
        "user_id",
        # 其他非特征信息（若存在则删）
        "data_name",
        "group_id",
        "person_day",
        "select_number",
        "total_select_number",
        "frequency_resolution",
        "cycle_number",
        "data_len",
        "_date_for_limit",
    ]

    def _process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "label" not in df.columns:
            raise ValueError("数据中没有 label 列")

        # 先删 forbidden
        df = df.drop(columns=[c for c in FORBIDDEN_COLS if c in df.columns], errors="ignore")

        # 删其他非特征列（但不删 age）
        df = df.drop(columns=[c for c in drop_not_features if c in df.columns], errors="ignore")

        # 构造 *_mag（全部 BRACKET）
        for col in BRACKET_FEATURE_COLS:
            cx = f"{col}_x"
            cy = f"{col}_y"
            if cx in df.columns and cy in df.columns:
                df[f"{col}_mag"] = np.sqrt(df[cx] ** 2 + df[cy] ** 2)

        # 删原始字符串列（如果还在）
        df = df.drop(columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns], errors="ignore")

        label = df["label"].reset_index(drop=True)
        age = df["age"].reset_index(drop=True) if "age" in df.columns else None

        # feature_df：排除 label/age/forbidden
        drop_cols = ["label", "age", *FORBIDDEN_COLS]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # 删 *_x/_y
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")],
            errors="ignore",
        )

        # 只保留数值列
        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"[deal_file] 初步特征数：{feature_df.shape[1]}")

        # 皮尔逊筛选（*_mag 全保留）
        if HIGH_CORR_FEATURES:
            keep_cols = []
            for col in feature_df.columns:
                if col.endswith("_mag"):
                    keep_cols.append(col)
                elif col in HIGH_CORR_FEATURES:
                    keep_cols.append(col)
            feature_df = feature_df[keep_cols]
            print(f"[deal_file] 皮尔逊后特征数：{feature_df.shape[1]}（*_mag 全保留）")
        else:
            print("[deal_file] 白名单为空：保留全部特征（含 *_mag）")

        # 最后确保 forbidden 不可能混入
        feature_df = feature_df.drop(columns=[c for c in FORBIDDEN_COLS if c in feature_df.columns], errors="ignore")

        if age is not None:
            out = pd.concat([age, feature_df.reset_index(drop=True), label], axis=1)
            # 列顺序：age ... label
            out.rename(columns={"age": "age"}, inplace=True)
        else:
            out = pd.concat([feature_df.reset_index(drop=True), label], axis=1)

        return out

    return _process(train_df), _process(test_df)


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

    # 断言：user_id 不存在
    leaked = [c for c in FORBIDDEN_COLS if c in train_df.columns or c in test_df.columns]
    if leaked:
        raise RuntimeError(f"❌ 泄漏列仍存在于输出特征中：{leaked}")

    train_df.to_csv(train_feat_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_feat_path, index=False, encoding="utf-8-sig")

    print(f"✅ 处理后的训练集已保存：{train_feat_path}")
    print(f"✅ 处理后的测试集已保存：{test_feat_path}")


if __name__ == "__main__":
    make_feature_csvs()
