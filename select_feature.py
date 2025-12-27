# select_feature.py
import os
import ast

import numpy as np
import pandas as pd

# ========= 基本路径配置 =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"

DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

# deal_data 输出的原始划分结果
TRAIN_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "train_raw_v5.csv")
TEST_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "test_raw_v5.csv")

# 本文件要输出的“特征处理后”的数据
TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v5.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v5.csv")


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

# 皮尔逊筛选阈值：|r| >= 0.15 且 Significant=True
PEARSON_ABS_CORR_THRESHOLD = 0.15
PEARSON_REQUIRE_SIGNIFICANT = True


def get_high_corr_feature_names(
    pearson_dir: str = PEARSON_DIR,
    files = PEARSON_FILES,
    abs_corr_thr: float = PEARSON_ABS_CORR_THRESHOLD,
    require_sig: bool = PEARSON_REQUIRE_SIGNIFICANT,
) -> set:
    """
    从多个皮尔逊分析结果 csv 中，选出“相关系数绝对值较大”的特征名集合。
    条件：|Correlation| >= abs_corr_thr 且（可选）Significant == True。
    """
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

    print(f"\n[Peasron] 合并后总共选中 {len(selected)} 个特征名\n")
    return selected


# 在模块加载时读皮尔逊结果，构建高相关特征白名单
HIGH_CORR_FEATURES = get_high_corr_feature_names()


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
    把所有 "[a, b]" 字符串列拆成两个数值列： col_x, col_y
    原始 col 列先保留，稍后在 deal_file 中统一删除。
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
    - 【新增】最后一步：只保留“皮尔逊相关系数较高”的特征
    - 返回：仅保留数值特征 + label
    """
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

        # 4. 分离 label 与 user_id（user_id 仅用于后续 GroupKFold 分组，不作为特征）
        label = df["label"]
        user_id = df["user_id"] if "user_id" in df.columns else None
        feature_df = df.drop(columns=[c for c in ["label", "user_id"] if c in df.columns])
        # 5. 删掉所有 *_x, *_y，只保留 *_mag 和其他数值列
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")],
            errors="ignore",
        )

        # 6. 删除所有 object 列（只保留数值特征）
        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"[deal_file] 初步处理后特征数：{feature_df.shape[1]}")

        # 7. 【关键】根据皮尔逊结果做特征筛选
        if HIGH_CORR_FEATURES:
            keep_cols = []
            for col in feature_df.columns:
                base = col
                # 对 *_mag，base 名去掉后缀 '_mag'，方便和皮尔逊里的 Feature 对应
                if col.endswith("_mag"):
                    base = col[:-4]
                if base in HIGH_CORR_FEATURES:
                    keep_cols.append(col)

            # 防止极端情况：如果一个都没匹配上，就保留全部特征
            if keep_cols:
                feature_df = feature_df[keep_cols]
                print(f"[deal_file] 皮尔逊筛选后保留特征数：{feature_df.shape[1]}")
            else:
                print("[deal_file] 警告：皮尔逊筛选没有匹配到任何特征，暂时保留全部特征。")
        else:
            print("[deal_file] 没有皮尔逊白名单（HIGH_CORR_FEATURES 为空），保留全部特征。")

        # 8. 重新拼回 user_id（便于后续 GroupKFold；训练时会 drop 掉它）
        if user_id is not None:
            df_processed = pd.concat([
                user_id.reset_index(drop=True),
                feature_df.reset_index(drop=True),
                label.reset_index(drop=True),
            ], axis=1)
        else:
            df_processed = pd.concat([feature_df, label], axis=1)
        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)

    return train_df, test_df


# ========= 生成特征 CSV =========

def make_feature_csvs(
    train_raw_path: str = TRAIN_RAW_PATH,
    test_raw_path: str = TEST_RAW_PATH,
    train_feat_path: str = TRAIN_FEAT_PATH,
    test_feat_path: str = TEST_FEAT_PATH,
):
    """
    从 deal_data 生成的 train_raw/test_raw 中：
      1) 展开 "[a,b]" 特征
      2) 删除非特征列，合成 *_mag，删掉 object 列
      3) 根据皮尔逊结果只保留高相关特征
      4) 保存为 train_vn.csv / test_vn.csv （包含 label）
    """
    # 读取 raw 划分结果（含所有原始列）
    train_df = pd.read_csv(train_raw_path)
    test_df = pd.read_csv(test_raw_path)

    print("原始训练集形状：", train_df.shape)
    print("原始测试集形状：", test_df.shape)

    # 展开 "[a,b]" 字符串列
    train_df = expand_bracket_features(train_df)
    test_df = expand_bracket_features(test_df)

    # 做特征清理 + 皮尔逊筛选
    train_df, test_df = deal_file(train_df, test_df)

    # 再保险处理一下 inf
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    # 保存为特征处理后的 CSV
    train_df.to_csv(train_feat_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_feat_path, index=False, encoding="utf-8-sig")

    print(f"\n处理后的训练集已保存到：{train_feat_path}")
    print(f"处理后的测试集已保存到：{test_feat_path}")


if __name__ == "__main__":
    # 单独运行本文件：直接生成 train.csv / test.csv
    make_feature_csvs()
