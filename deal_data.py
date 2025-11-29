# deal_data.py
import pandas as pd
import numpy as np
import ast

# 这些列都是 "[a, b]" 格式的字符串，需要拆成两列
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


def safe_eval(val):
    """安全解析 '[a, b]' 字符串，异常时返回 [None, None]"""
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
    原始 col 列保留不动（稍后在 deal_file 里统一删）
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
    针对新的 5-sheet 特征表：
    - 去掉所有不参与训练的标识列
    - 删除所有 "[a,b]" 的原始字符串列（FDA1、frequency_lobe*、ppg_peak...）
    - 只保留数值特征 + label
    """

    # 各个 sheet 里不参与训练的“标识/索引类”列
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

        # 1. 删掉各种非特征列
        df = df.drop(
            columns=[c for c in drop_not_features if c in df.columns],
            errors="ignore"
        )

        # 2. 删掉所有 "[a,b]" 的原始字符串列
        df = df.drop(
            columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns],
            errors="ignore"
        )

        # 3. 只保留数值特征 + label
        if "label" not in df.columns:
            raise ValueError("数据中没有 label 列，请确认在调用 deal_file 前已经打好标签。")

        label = df["label"]
        feature_df = df.drop(columns=["label"])

        # 只保留数值型特征列
        feature_df = feature_df.select_dtypes(include=[np.number])

        # 拼回 label
        df_processed = pd.concat([feature_df, label], axis=1)

        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)

    return train_df, test_df


def load_one_user_file(path: str) -> pd.DataFrame:
    """
    读取一个用户的 xlsx：
    - 读 5 个 sheet
    - 用公共键合并
    注意：不在这里展开 [a,b]，在主流程对 df_all 统一展开。
    """
    df_prv = pd.read_excel(path, sheet_name="feature_prv")
    df_time = pd.read_excel(path, sheet_name="feature_time")
    df_welch = pd.read_excel(path, sheet_name="feature_welch")
    df_ref = pd.read_excel(path, sheet_name="reference_time")
    df_freq = pd.read_excel(path, sheet_name="feature_frequency")

    def smart_merge(left, right):
        cand_keys = [
            "user_id", "data_name",
            "cycle_number", "select_number", "total_select_number"
        ]
        keys = [k for k in cand_keys if k in left.columns and k in right.columns]
        drop_cols = [c for c in right.columns if c in left.columns and c not in keys]
        right2 = right.drop(columns=drop_cols)
        return left.merge(right2, on=keys, how="left")

    base = df_prv
    base = smart_merge(base, df_time)
    base = smart_merge(base, df_welch)
    base = smart_merge(base, df_ref)
    base = smart_merge(base, df_freq)

    return base
