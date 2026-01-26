import argparse
import math
import numpy as np
import pandas as pd


# ========= 疾病判定（沿用旧逻辑） =========

def has_disease_value(x) -> bool:
    if pd.isna(x):
        return False
    x = str(x).strip()
    if x in ("0", "无", "否", "", "nan", "NaN"):
        return False
    return True


DISEASE_COLS = [
    "某些感染性疾病或寄生虫病",
    "肿瘤",
    "血液或造血器官疾病",
    "免疫系统疾病",
    "内分泌、营养或代谢疾病",
    "精神、行为或神经发育障碍",
    "睡眠-觉醒障碍",
    "精神系统疾病",
    "循环系统疾病",
    "呼吸系统疾病",
    "消化系统疾病",
    "肌肉骨骼系统或结缔组织系统疾病",
    "泌尿生殖系统疾病",
]


def apply_disease_filter(user_info_row: pd.Series, age: int) -> bool:
    return True
    has_any_disease = any(has_disease_value(user_info_row[col]) for col in DISEASE_COLS if col in user_info_row.index) 
    has_endocrine = has_disease_value(user_info_row.get("内分泌、营养或代谢疾病", np.nan)) 
    has_circulatory = has_disease_value(user_info_row.get("循环系统疾病", np.nan)) 
    has_digestive = has_disease_value(user_info_row.get("消化系统疾病", np.nan))
    
    if age <= 50:
        return not has_any_disease
    else:
        return not has_any_disease
    

# ========= 年龄段（10岁窗宽） =========

def age_to_decade_bin(age: int) -> int:
    return int(age // 10) * 10


def decade_to_label(decade_start: int) -> str:
    return f"{decade_start}-{decade_start+9}"


# ========= 划分 + 重采样 =========

def stratified_split_70_30(df: pd.DataFrame, group_col: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for _, g in df.groupby(group_col):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(math.floor(n * 0.7))
        train_indices.extend(idx[:n_train].tolist())
        test_indices.extend(idx[n_train:].tolist())

    return train_indices, test_indices


def mean_impute_train_apply(df_train: pd.DataFrame, df_test: pd.DataFrame, exclude_cols: set):
    feature_cols = [c for c in df_train.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])]
    means = df_train[feature_cols].mean(numeric_only=True)

    df_train2 = df_train.copy()
    df_test2 = df_test.copy()

    df_train2[feature_cols] = df_train2[feature_cols].fillna(means)
    df_test2[feature_cols] = df_test2[feature_cols].fillna(means)

    return df_train2, df_test2, feature_cols


def resample_by_age_group_with_noise(
    df_train: pd.DataFrame,
    age_group_col: str,
    feature_cols: list,
    seed: int = 42,
    noise_scale: float = 0.01,
):
    """
    训练集内部按年龄段均衡（上采样 + 下采样）：
      - target_n = 各组样本量的最大值
      - n < target_n：保留原样本 + 有放回上采样到 target_n，并对连续特征加噪声
      - n > target_n：无放回下采样到 target_n
      - n == target_n：原样保留
    """
    rng = np.random.default_rng(seed)

    counts = df_train[age_group_col].value_counts().sort_index()
    target_n = int(counts.max())
    target_n = max(target_n, 1)

    # 每个特征列的标准差，用于噪声强度（σ = noise_scale * σ_col）
    col_stds = df_train[feature_cols].std(numeric_only=True).replace(0, np.nan)

    parts = []
    for group_value, g in df_train.groupby(age_group_col):
        n = len(g)

        # 1) 多数类：无放回下采样到 target_n
        if n > target_n:
            sampled_down = g.sample(
                n=target_n,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            parts.append(sampled_down)
            continue

        # 2) 刚好：原样保留
        if n == target_n:
            parts.append(g)
            continue

        # 3) 少数类：保留原样本 + 有放回补齐到 target_n，并加噪声
        need = target_n - n
        sampled_up = g.sample(
            n=need,
            replace=True,
            random_state=int(rng.integers(0, 1_000_000_000)),
        ).copy()

        # 加噪声（只对 feature_cols）
        for c in feature_cols:
            std = col_stds.get(c, np.nan)
            if pd.isna(std) or std == 0:
                continue
            sigma = noise_scale * float(std)
            eps = rng.normal(loc=0.0, scale=sigma, size=need)
            sampled_up[c] = sampled_up[c].astype(float) + eps

        parts.append(pd.concat([g, sampled_up], ignore_index=True))

    df_balanced = pd.concat(parts, ignore_index=True)
    return df_balanced, target_n



def main():
    parser = argparse.ArgumentParser(description="User-level 数据过滤 + 年龄段分层划分(7/3) + 训练集年龄段重采样（仅上采样，含噪声）")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--user_info_path", type=str, required=True)
    parser.add_argument("--train_csv_out", type=str, required=True)
    parser.add_argument("--test_csv_out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_age_min", type=int, default=20)
    parser.add_argument("--test_age_max", type=int, default=80)
    parser.add_argument("--noise_scale", type=float, default=0.01)
    args = parser.parse_args()

    # 读入 user 级特征表
    df = pd.read_csv(args.input_csv)
    if "user_id" not in df.columns:
        raise ValueError("input_csv 必须包含 user_id 列（每个 user 一行）")
    df["user_id"] = df["user_id"].astype(str).str.lstrip("_")

    
    # ★ 非特征列关键词（只要列名包含任意一个就删）
    NON_FEATURE_KEYWORDS = [
        "data_name",
        "group_id",
        "person_day",
        "select_number",
        "total_select_number",
        "frequency_resolution",
        "cycle_number",
        "data_len",
        "version"]

    # ★ 按“列名包含关键词”删除非特征列
    cols_to_drop = [
        c for c in df.columns
        if any(k in c for k in NON_FEATURE_KEYWORDS)
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")


    # 读用户信息（年龄 + 疾病）
    user_info = pd.read_csv(args.user_info_path)
    user_info["user_id"] = user_info["user_id"].astype(str).str.lstrip("_")

    # ===== 合并年龄（以 user_info 为准，防止 年龄_x / 年龄_y） =====

    # 如果 input_csv 里已经有“年龄”，先删掉，避免 merge 后重名
    if "年龄" in df.columns:
        df = df.drop(columns=["年龄"])

    if "年龄" not in user_info.columns:
        raise ValueError("user_info_path 中必须包含 '年龄' 列")

    user_age = user_info[["user_id", "年龄"]].copy()
    user_age["年龄"] = user_age["年龄"].astype(int)

    df = df.merge(user_age, on="user_id", how="inner")

    # 防御式处理：确保最终 df 一定有“年龄”列
    if "年龄" not in df.columns:
        if "年龄_y" in df.columns:
            df = df.rename(columns={"年龄_y": "年龄"})
            if "年龄_x" in df.columns:
                df = df.drop(columns=["年龄_x"])
        elif "年龄_x" in df.columns:
            df = df.rename(columns={"年龄_x": "年龄"})


    # 疾病过滤（按 user）
    user_info_idx = user_info.set_index("user_id")
    keep_mask = []
    for uid, age in zip(df["user_id"].tolist(), df["年龄"].tolist()):
        if uid not in user_info_idx.index:
            keep_mask.append(False)
            continue
        keep_mask.append(apply_disease_filter(user_info_idx.loc[uid], int(age)))

    df = df.loc[keep_mask].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("疾病过滤后没有任何用户，请检查过滤逻辑或输入数据。")

    # 年龄段（10岁窗宽）
    df["年龄段起点"] = df["年龄"].apply(age_to_decade_bin)
    df["年龄段"] = df["年龄段起点"].apply(decade_to_label)

    # 定义测试域：20–80
    in_test_domain = (df["年龄"] >= args.test_age_min) & (df["年龄"] <= args.test_age_max)
    test_domain = df.loc[in_test_domain].copy()
    non_test_domain = df.loc[~in_test_domain].copy()

    # 在测试域内按年龄段 7/3 分层切分
    train_idx, test_idx = stratified_split_70_30(test_domain, group_col="年龄段", seed=args.seed)
    test_domain_train = test_domain.loc[train_idx].copy()
    test_df = test_domain.loc[test_idx].copy()

    # 训练域包含：非测试域全部 + 测试域内的 70%
    train_df = pd.concat([non_test_domain, test_domain_train], ignore_index=True)

    # 不输出类别 label（确保删掉 label 列）
    if "label" in train_df.columns:
        train_df = train_df.drop(columns=["label"])
    if "label" in test_df.columns:
        test_df = test_df.drop(columns=["label"])

    # 缺失值均值插补（仅训练估计）
    exclude_cols_for_impute = {"user_id", "年龄", "年龄段", "年龄段起点"}
    train_df_imp, test_df_imp, feature_cols = mean_impute_train_apply(
        train_df, test_df, exclude_cols=exclude_cols_for_impute
    )

    train_balanced, target_n = resample_by_age_group_with_noise(
        train_df_imp,
        age_group_col="年龄段",
        feature_cols=feature_cols,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )


    # 输出列顺序：user_id, 年龄, (特征列...)
    keep_cols = ["user_id", "年龄"] + feature_cols
    train_out = train_balanced[keep_cols]
    test_out = test_df_imp[keep_cols]

    train_out.to_csv(args.train_csv_out, index=False, encoding="utf-8-sig")
    test_out.to_csv(args.test_csv_out, index=False, encoding="utf-8-sig")

    print("✅ 完成")
    print(f"训练集（重采样后）行数：{len(train_out)} | 用户数（user_id去重）：{train_out['user_id'].nunique()}")
    print(f"测试集（无扰动）行数：{len(test_out)} | 用户数：{test_out['user_id'].nunique()}")
    print(f"训练集上采样目标 target_n（各年龄段对齐到）：{target_n}")


if __name__ == "__main__":
    main()
