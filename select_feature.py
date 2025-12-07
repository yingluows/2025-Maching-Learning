# select_feature.py
import os
import ast

import numpy as np
import pandas as pd

# ========= 路径配置 =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"

DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(DATA_SPLIT_DIR, "train_raw.csv")
TEST_CSV_PATH = os.path.join(DATA_SPLIT_DIR, "test_raw.csv")

# ========= 特征列配置（含 "[a,b]" 字符串列） =========
BRACKET_FEATURE_COLS = [
    # feature_time 里的 FDA/FDB
    "FDA1", "FDB1",

    # feature_welch 里的 frequency_lobe*
    *[f"frequency_lobe{i}" for i in range(11)],

    # reference_time 里的 ppg / rising / falling 等
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


# ========= 特征处理相关函数 =========

def safe_eval(val):
    """把 '[a, b]' 这类字符串安全地转成 list；错误时返回 [None, None]。"""
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        else:
            return [None, None]
    except Exception:
        return [None, None]


def expand_bracket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    把所有 "[a, b]" 字符串列拆成 两个 数值列： col_x, col_y
    原始 col 列暂时保留（在 deal_file 里统一删）
    """
    df = df.copy()
    for col in BRACKET_FEATURE_COLS:
        if col in df.columns:
            two_cols = df[col].apply(safe_eval).apply(pd.Series)
            df[f"{col}_x"] = two_cols[0]
            df[f"{col}_y"] = two_cols[1]
    return df


def deal_file(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    - 删除各种非特征列（user_id / data_name / person_day / select_number 等）
    - 对 BRACKET_FEATURE_COLS 的 *_x, *_y 构造新特征 col_mag = sqrt(x^2 + y^2)
    - 删掉原始的字符串列 + *_x + *_y
    - 删除所有 object 列
    - 返回：仅保留数值特征 + label
    """
    drop_not_features = [
        "user_id",
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
            raise ValueError("数据中没有 label 列，请确认在调用 deal_file 前已经打好标签。")

        # 1. 删各种非特征列
        df = df.drop(
            columns=[c for c in drop_not_features if c in df.columns],
            errors="ignore",
        )

        # 2. 用 *_x, *_y 构造合成特征 col_mag
        for col in BRACKET_FEATURE_COLS:
            cx = f"{col}_x"
            cy = f"{col}_y"
            if cx in df.columns and cy in df.columns:
                df[f"{col}_mag"] = np.sqrt(df[cx] ** 2 + df[cy] ** 2)

        # 3. 删掉原始 "[a,b]" 字符串列
        df = df.drop(
            columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns],
            errors="ignore",
        )

        # 4. 分离 label
        label = df["label"]
        feature_df = df.drop(columns=["label"])

        # 5. 删掉所有 *_x, *_y，只保留 *_mag 和其他数值列
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")],
            errors="ignore",
        )

        # 6. 删除所有 object 列（只保留数值特征）
        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"[deal_file] 处理后保留特征数：{feature_df.shape[1]}")

        df_processed = pd.concat([feature_df, label], axis=1)
        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)

    return train_df, test_df


# ========= 对外主函数：给 main 使用 =========

def load_and_process_data(
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH,
):
    """
    从 deal_data 生成的 train_raw/test_raw 中：
      1) 展开 "[a,b]" 特征
      2) 删除非特征列，合成 *_mag，删掉 object 列
      3) 返回 X_train, X_test, y_train, y_test
    """
    # 读取 raw 划分结果（含所有原始列）
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("原始训练集形状：", train_df.shape)
    print("原始测试集形状：", test_df.shape)

    # 展开 "[a,b]" 字符串列
    train_df = expand_bracket_features(train_df)
    test_df = expand_bracket_features(test_df)

    # 统一做特征清理
    train_df, test_df = deal_file(train_df, test_df)

    # 分离 X/y
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # 替换 inf 为 NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    print("处理后训练集形状：", X_train.shape)
    print("处理后测试集形状：", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 简单自测：只跑特征处理，看看形状是否正常
    X_train, X_test, y_train, y_test = load_and_process_data()
    print("X_train columns:", len(X_train.columns))
