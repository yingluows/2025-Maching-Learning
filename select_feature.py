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
TRAIN_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "train_all.csv")
TEST_RAW_PATH = os.path.join(DATA_SPLIT_DIR, "test_all.csv")

# 本文件输出
TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_all_top.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_all_top.csv")

LABEL_TO_RANGE = {
    0: (20, 29),
    1: (30, 39),
    2: (40, 49),
    3: (50, 59),
    4: (60, 69),
    5: (70, 79),
}

def age_to_label(age):
    if pd.isna(age):
        return np.nan
    for label, (low, high) in LABEL_TO_RANGE.items():
        if low <= age <= high:
            return label
    return np.nan


# ========= 皮尔逊结果配置 =========
PEARSON_DIR = os.path.join(BASE_DIR, "PEARSON_DIR")

# 只用这四份来为 ratio(0~10) 从 welch+frequency 中挑“相关性最高”
RATIO_PEARSON_FILES = [
    "Feature_welch_1皮尔逊分析.csv",
    "Feature_welch_2皮尔逊分析.csv",
    "全皮尔逊分析Feature_frequency_1.csv",
    "全皮尔逊分析Feature_frequency_2.csv",
]
PEARSON_REQUIRE_SIGNIFICANT = True

# ========= 你要最终保留的“基准特征名” =========
TIME_FEATURES = ["BAR", "FDA3", "FDB3", "AGI", "AUI", "LASI"]
PRV_FEATURES  = ["pnn20", "nn20", "HF", "sd1_sd2_ratio"]
RATIO_BASES   = [f"lobe_spectrum_ratio{i}" for i in range(11)]

# ========= 特征列配置（含 "[a,b]" 字符串列） =========
# 你这份 raw 数据里这类列一般是 "[a,b]" 形式，需要先拆成 *_x *_y
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


# ========= 工具函数：读取皮尔逊表、匹配候选、挑 |corr| 最大 =========
def load_corr_table(pearson_dir: str, files: list, require_sig: bool = True) -> pd.DataFrame:
    frames = []
    for fname in files:
        path = os.path.join(pearson_dir, fname)
        if not os.path.exists(path):
            print(f"⚠ 未找到皮尔逊文件: {path}，跳过")
            continue

        df = pd.read_csv(path)
        if "Feature" not in df.columns or "Correlation" not in df.columns:
            print(f"⚠ 文件 {path} 缺少 Feature/Correlation 列，跳过")
            continue

        df = df.copy()
        df["Feature"] = df["Feature"].astype(str)
        df["abs_corr"] = df["Correlation"].abs()

        if require_sig and "Significant" in df.columns:
            df = df[df["Significant"] == True]

        frames.append(df[["Feature", "Correlation", "abs_corr"]])

    if not frames:
        return pd.DataFrame(columns=["Feature", "Correlation", "abs_corr"])

    out = pd.concat(frames, ignore_index=True)
    # 同名 Feature 出现多次：保留 |corr| 最大
    out = out.sort_values("abs_corr", ascending=False).drop_duplicates("Feature", keep="first")
    return out


def find_candidates_by_base(base: str, all_features: list[str]) -> list[str]:
    """
    在真实数据列名里找 base 的候选：
    - 精确等于 base
    - 或以 base 结尾（适配 prefix__base / xxx_base 等）
    """
    cands = []
    for f in all_features:
        if f == base or f.endswith(base):
            cands.append(f)
    return cands


def pick_best_by_corr(base: str, candidates: list[str], corr_df: pd.DataFrame) -> str | None:
    """
    从候选列里选 |corr| 最大的一列。
    corr_df 的 Feature 可能与列名完全一致，也可能只记录 base；
    这里用 endswith 做一次“松匹配”。
    """
    if not candidates:
        return None

    if corr_df.empty:
        return base if base in candidates else candidates[0]

    corr_map = dict(zip(corr_df["Feature"], corr_df["abs_corr"]))

    def score(col: str) -> float:
        if col in corr_map:
            return corr_map[col]
        # 松匹配：col 以 corr表特征结尾 or corr表特征以 col 结尾
        for k, v in corr_map.items():
            if col.endswith(k) or k.endswith(col):
                return v
        return -1.0

    best = max(candidates, key=score)
    return best if score(best) >= 0 else (base if base in candidates else candidates[0])


# 只用于 ratio 的相关性选择（welch+frequency 合并）
RATIO_CORR_DF = load_corr_table(PEARSON_DIR, RATIO_PEARSON_FILES, require_sig=PEARSON_REQUIRE_SIGNIFICANT)


# ========= bracket 解析 =========
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
    新规则（你要求：去掉 _mag 特征）：
    - 不再构造 *_mag
    - 删除 bracket 原始列("[a,b]")，并删除 *_x/*_y
    - 最终只保留：
        时域: BAR, FDA3, FDB3, AGI, AUI, LASI
        PRV : pnn20, nn20, HF, sd1_sd2_ratio
        频域: lobe_spectrum_ratio0~10（从 welch+frequency 候选里取 |corr| 最大）
    - 删除非特征列与 user_id
    """

    drop_not_features = [
        #"user_id",
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

        # 1) 删除非特征列
        df = df.drop(columns=[c for c in drop_not_features if c in df.columns], errors="ignore")

        # 2) 删除原始 bracket 列（"[a,b]"）
        df = df.drop(columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns], errors="ignore")

        # 3) 分离 label
       
        y = df["label"]

        meta_cols = []
        for c in ["user_id", "年龄"]:
            if c in df.columns:
                meta_cols.append(c)

        meta_df = df[meta_cols].copy()  # ✅ 元信息

        X = df.drop(columns=["label"] + meta_cols)

        # 4) 删除 *_x, *_y（你不想要这些，也不构造 _mag）
        X = X.drop(columns=[c for c in X.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")

        # 5) 仅保留数值列（避免混入 object）
        X = X.select_dtypes(include=[np.number])

        # 6) 按你的名单挑选
        all_cols = list(X.columns)

        # (A) 时域 & PRV 固定保留（若多版本则用相关性挑一个；这里使用 RATIO_CORR_DF 作为“可用相关性表”）
        fixed_bases = TIME_FEATURES + PRV_FEATURES
        fixed_keep = []
        for base in fixed_bases:
            cands = find_candidates_by_base(base, all_cols)
            best = pick_best_by_corr(base, cands, RATIO_CORR_DF)
            if best:
                fixed_keep.append(best)

        # (B) ratio0~10：全都取，但排除 feature_frequency_1 的版本
        ratio_keep = []
        for base in RATIO_BASES:
            cands = find_candidates_by_base(base, all_cols)

            # 排除 feature_frequency_1 的候选（大小写不敏感）
            cands = [c for c in cands if "feature_frequency_1" not in c.lower()]

            if not cands:
                print(f"⚠ {base} 没有可用候选（可能全被 feature_frequency_1 过滤了，或数据里本来就没有）")
                continue

            ratio_keep.extend(cands)  # ✅ 用 extend，避免 list 嵌套导致 unhashable

        # 合并去重
        keep_cols = fixed_keep + ratio_keep
        seen = set()
        keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
        keep_cols = [c for c in keep_cols if c in X.columns]

        X = X[keep_cols]
        print(
            f"[deal_file] 固定时域+PRV={len(fixed_keep)}；ratio(0~10)={len(ratio_keep)}；总={X.shape[1]}"
        )

        out = pd.concat([meta_df, X], axis=1)
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

    # ===== 生成 label（由 age 映射）=====
    if "label" not in train_df.columns:
        if "年龄" not in train_df.columns:
            raise ValueError("未找到 age 列，无法生成 label，请确认原始数据列名。")

        train_df["label"] = train_df["年龄"].apply(age_to_label)
        test_df["label"] = test_df["年龄"].apply(age_to_label)

    # 可选：删除 age，避免数据泄漏
    # train_df = train_df.drop(columns=["age"])
    # test_df = test_df.drop(columns=["age"])

    print("✔ 已根据 age 成功生成 label 列")


    print("原始训练集形状：", train_df.shape)
    print("原始测试集形状：", test_df.shape)

    # bracket 展开（虽然最终会删 *_x/_y，但不展开的话可能会影响你后续检查/兼容旧流程）
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
