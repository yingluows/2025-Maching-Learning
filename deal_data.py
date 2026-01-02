# deal_data.py
"""
数据划分脚本（v7策略）
- 不再随机抽取样本：对每个用户的每一天取特征均值（daily mean）
- 保留 50 岁疾病过滤规则：
    * age <= 50：必须无任何疾病
    * age  > 50：必须有“循环系统疾病”
- 训练集/测试集按【人数】划分（按 label 分层），并保留 user_id（用于可追溯/按人划分）
- 输出为原始划分后的 CSV：train_raw_v7.csv / test_raw_v7.csv
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ========= 路径配置（根据需要修改） =========
FEATURE_DIR = r"D:/2025_Stage/Code/XGB/ppgfeature_v114"
USER_INFO_PATH = r"D:/2025_Stage/Code/XGB/用户疾病分类统计.csv"

OUTPUT_DIR = r"D:/2025_Stage/Code/XGB/Data_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "train_raw_v7.csv")
TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "test_raw_v7.csv")

# 每天只保留 cycle 数最多的前 N 个 data_name（保留你原来的逻辑；不想要可改成 None）
TOP_DATA_NAME_PER_DAY = 8


# ========= 标签分段 =========
def age_to_group(age: int) -> int:
    age = int(age)
    if age <= 20:
        return 0
    elif age <= 30:
        return 1
    elif age <= 40:
        return 2
    elif age <= 50:
        return 3
    elif age <= 60:
        return 4
    else:
        return 5


# ========= 日期解析 =========
def ensure_date_column(df: pd.DataFrame, date_col: str = "_date_for_limit") -> pd.DataFrame:
    """
    确保存在一个表示“日期”的列（YYYY-MM-DD 字符串），从 data_name 中解析：
    data_name 形如 xxx_YYYYMMDDhhmmss，取中间 YYYYMMDD。
    """
    df = df.copy()
    if date_col in df.columns:
        return df

    if "data_name" not in df.columns:
        raise ValueError("缺少 data_name 列，无法解析日期。")

    def _parse_date(x: str) -> str:
        s = str(x)
        # 尝试从字符串中提取 8 位日期
        # 常见：xxx_20240102123000 或 xxx-20240102123000
        import re
        m = re.search(r"(20\d{6})", s)
        if not m:
            return "1970-01-01"
        ymd = m.group(1)
        return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"

    df[date_col] = df["data_name"].apply(_parse_date)
    return df


# ========= 每天取 cycle 数最多的前 N 个 data_name =========
def filter_top_data_names_per_day(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    对每个用户、每一天，统计每个 data_name 的 cycle_number 行数（近似 cycle 数），
    取 cycle 数最多的前 top_n 个 data_name，保留对应行。
    """
    if top_n is None:
        return df

    if "data_name" not in df.columns or "cycle_number" not in df.columns:
        print("⚠ 缺少 data_name/cycle_number，无法按每天前 N 个 data_name 过滤，原样返回。")
        return df

    df = ensure_date_column(df, date_col="_date_for_limit")

    # (user, date, data_name) -> count
    grp = (
        df.groupby(["user_id", "_date_for_limit", "data_name"])["cycle_number"]
        .count()
        .reset_index(name="cycle_count")
    )

    # 每个 (user,date) 取 top_n 个 data_name
    grp = grp.sort_values(["user_id", "_date_for_limit", "cycle_count"], ascending=[True, True, False])
    top = grp.groupby(["user_id", "_date_for_limit"]).head(top_n)

    keep = df.merge(
        top[["user_id", "_date_for_limit", "data_name"]],
        on=["user_id", "_date_for_limit", "data_name"],
        how="inner",
    )
    return keep


# ========= 合并一个用户的多 sheet 特征 =========
def load_one_user_file(path: str) -> pd.DataFrame:
    """
    你的特征文件如果是 Excel（多 sheet），这里按你原来的 sheet 名读取并合并。
    如果你的文件结构不同，可以在这里调整。
    """
    df_prv = pd.read_excel(path, sheet_name="feature_prv")
    df_time = pd.read_excel(path, sheet_name="feature_time")
    df_welch1 = pd.read_excel(path, sheet_name="feature_welch_1")
    df_welch2 = pd.read_excel(path, sheet_name="feature_welch_2")
    df_ref = pd.read_excel(path, sheet_name="reference_time")
    df_freq1 = pd.read_excel(path, sheet_name="feature_frequency_1")
    df_freq2 = pd.read_excel(path, sheet_name="feature_frequency_2")

    def smart_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        cand_keys = ["user_id", "data_name", "cycle_number", "select_number", "total_select_number"]
        keys = [k for k in cand_keys if k in left.columns and k in right.columns]
        if not keys:
            # 找不到共同 key，就按 index 连接（不推荐，但兜底）
            return pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)
        return pd.merge(left, right, on=keys, how="inner")

    base = smart_merge(df_prv, df_time)
    base = smart_merge(base, df_welch1)
    base = smart_merge(base, df_welch2)
    base = smart_merge(base, df_ref)
    base = smart_merge(base, df_freq1)
    base = smart_merge(base, df_freq2)
    return base


# ========= 疾病判断 =========
def has_disease_value(x) -> bool:
    """
    判断一个疾病单元格是否表示“有疾病”
    兼容：0/1、是/否、有/无、文本、NaN
    """
    if pd.isna(x):
        return False
    x = str(x).strip()
    if x in ("0", "无", "否", "", "nan", "NaN"):
        return False
    return True


def daily_mean_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个用户的每一天取均值（只对数值列）。
    保留列：
      - user_id
      - _date_for_limit
      - age, label（在进入该函数前已填好）
      - 其他数值列：mean
    """
    df = ensure_date_column(df, date_col="_date_for_limit")

    # 保证 age/label 存在（应当恒定）
    if "age" not in df.columns or "label" not in df.columns:
        raise ValueError("daily_mean_aggregate 需要 df 中包含 age 和 label。")

    group_cols = ["user_id", "_date_for_limit"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 但 user_id/date 不在 numeric；age/label 在 numeric，均值不变
    agg = df.groupby(group_cols, as_index=False)[numeric_cols].mean()

    # 把 age/label 保持为 int（均值后可能是 float）
    agg["age"] = agg["age"].round().astype(int)
    agg["label"] = agg["label"].round().astype(int)

    return agg


def split_by_person_stratified_no_sampling(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按人分层划分（不做 per-class 样本抽取），保证 train/test 的 user 不重叠。
    """
    rng = np.random.RandomState(seed)

    user_label = (
        df.groupby(user_id_col)[label_col]
        .first()
        .reset_index()
    )

    train_users = []
    test_users = []

    for lab, sub in user_label.groupby(label_col):
        users = sub[user_id_col].tolist()
        rng.shuffle(users)
        n_train = int(round(len(users) * train_person_ratio))
        train_users.extend(users[:n_train])
        test_users.extend(users[n_train:])

    train_df = df[df[user_id_col].isin(train_users)].reset_index(drop=True)
    test_df = df[df[user_id_col].isin(test_users)].reset_index(drop=True)

    # 安全检查
    inter = set(train_users).intersection(set(test_users))
    if inter:
        raise RuntimeError(f"train/test 用户集合有重叠：{len(inter)} 个")

    return train_df, test_df


def prepare_and_save_splits(
    feature_dir: str = FEATURE_DIR,
    user_info_path: str = USER_INFO_PATH,
    train_csv_path: str = TRAIN_CSV_PATH,
    test_csv_path: str = TEST_CSV_PATH,
):
    user_info = pd.read_csv(user_info_path)
    user_info["user_id"] = user_info["user_id"].astype(str).str.lstrip("_")

    # 疾病列：除 user_id、年龄 外的所有列都视为疾病字段（按你的表结构）
    possible_non_disease = {"user_id", "年龄"}
    disease_cols = [c for c in user_info.columns if c not in possible_non_disease]

    all_daily = []

    # 收集文件
    feature_files: List[str] = []
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.xlsx")))
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.csv")))
    feature_files = sorted(feature_files)

    for path in tqdm(feature_files, desc="正在加载用户特征数据"):
        name = os.path.basename(path)
        if name.startswith("~$"):
            continue

        user_id_from_name = os.path.splitext(os.path.basename(path))[0]

        # 读取特征
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".xlsx"):
            df = load_one_user_file(path)
        else:
            continue

        # 确保 user_id
        if "user_id" in df.columns:
            df["user_id"] = df["user_id"].astype(str).str.lstrip("_")
            user_id_in_file = str(df["user_id"].iloc[0])
            user_id_from_name = user_id_in_file
        else:
            df["user_id"] = str(user_id_from_name).lstrip("_")

        uid = str(user_id_from_name).lstrip("_")

        # 查用户信息
        row = user_info[user_info["user_id"] == uid]
        if row.empty:
            # 找不到用户信息就跳过
            continue

        age = int(row["年龄"].iloc[0])
        label = age_to_group(age)

        # ===== 50岁疾病过滤规则 =====
        has_any_disease = any(has_disease_value(row[c].iloc[0]) for c in disease_cols)
        has_circulatory = has_disease_value(row["循环系统疾病"].iloc[0]) if "循环系统疾病" in row.columns else False

        if age <= 50:
            # 必须完全无疾病
            if has_any_disease:
                continue
        else:
            # 必须有循环系统疾病
            if not has_circulatory:
                continue

        # 写入 age/label
        df["age"] = age
        df["label"] = label

        # 保留你原来的 data_name top 过滤
        if TOP_DATA_NAME_PER_DAY is not None:
            df = filter_top_data_names_per_day(df, top_n=TOP_DATA_NAME_PER_DAY)

        if df.empty:
            continue

        # 每天取均值（不再随机采样）
        df_daily = daily_mean_aggregate(df)
        all_daily.append(df_daily)

    if not all_daily:
        raise RuntimeError("没有成功读取到任何用户数据，请检查特征路径和用户列表/过滤条件。")

    df_all = pd.concat(all_daily, ignore_index=True)

    print("\n=== 汇总完成（daily mean）===")
    print("总样本数：", len(df_all))
    print("总人数：", df_all["user_id"].nunique())
    print("各类样本数：\n", df_all["label"].value_counts().sort_index())

    # 按人分层划分（不抽样）
    train_df, test_df = split_by_person_stratified_no_sampling(
        df_all,
        user_id_col="user_id",
        label_col="label",
        train_person_ratio=0.8,
        seed=42,
    )

    print("\n=== 划分完成 ===")
    print("训练集样本数：", len(train_df), "人数：", train_df["user_id"].nunique())
    print("测试集样本数：", len(test_df), "人数：", test_df["user_id"].nunique())

    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 训练集已保存：{train_csv_path}")
    print(f"✅ 测试集已保存：{test_csv_path}")


if __name__ == "__main__":
    prepare_and_save_splits()
