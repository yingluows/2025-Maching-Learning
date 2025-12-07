# deal_data.py
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from split import split_by_person_stratified


# ========= 路径配置（根据需要修改） =========
FEATURE_DIR = r"D:/2025_Stage/Code/XGB/ppgfeature_v114"
USER_INFO_PATH = r"D:/2025_Stage/Code/XGB/用户列表.csv"

OUTPUT_DIR = r"D:/2025_Stage/Code/XGB/Data_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出的是“原始划分后的数据”，不做特征处理
TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "train_raw.csv")
TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "test_raw.csv")

# 每人每天最多保留的样本数
MAX_PER_DAY_READ = 30


# ========= 通用函数 =========

def age_to_group(age: int) -> int:
    """把年龄映射到年龄段 label。"""
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


def ensure_date_column(df: pd.DataFrame, date_col: str = "_date_for_limit") -> pd.DataFrame:
    """
    确保存在一个表示“日期”的列（YYYY-MM-DD 字符串），从 data_name 中解析：
    data_name 形如 xxx_YYYYMMDDhhmmss，取中间 YYYYMMDD。
    """
    df = df.copy()
    if date_col in df.columns:
        return df

    if "data_name" in df.columns:
        ts_str = df["data_name"].astype(str).str.extract(r"_(\d{8})\d{6}$", expand=False)
        dates = pd.to_datetime(ts_str, format="%Y%m%d", errors="coerce")
        df[date_col] = dates.dt.strftime("%Y-%m-%d")
    else:
        df[date_col] = "NA"

    return df


def filter_top_data_names_per_day(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    对单个用户的数据：
      - 先按日期 + data_name 统计 cycle 数量（行数）；
      - 对每个“日期”，选出 cycle 数最多的前 top_n 个 data_name；
      - 只保留这些 (日期, data_name) 对应的所有行；
      - 如果某天 data_name 少于 top_n，则全保留；
      - 如果某天压根就没记录，自然就“跳过那一天”。

    之后再做按天 max_per_day 采样。
    """
    if "data_name" not in df.columns or "cycle_number" not in df.columns:
        print("⚠ 警告：数据中不存在 'data_name' 或 'cycle_number' 列，无法按每天前 N 个 data_name 过滤，原样返回。")
        return df

    df = ensure_date_column(df, date_col="_date_for_limit")

    # 统计每个 (日期, data_name) 的 cycle 数（这里用行数即可）
    group = (
        df.groupby(["_date_for_limit", "data_name"])["cycle_number"]
          .count()
          .reset_index(name="cycle_count")
    )

    # 对每一天，取 cycle_count 最大的前 top_n 个 data_name
    group_sorted = group.sort_values(["_date_for_limit", "cycle_count"],
                                     ascending=[True, False])
    top = group_sorted.groupby("_date_for_limit").head(top_n)

    # 只保留这些 (日期, data_name) 的行
    df_filtered = df.merge(
        top[["_date_for_limit", "data_name"]],
        on=["_date_for_limit", "data_name"],
        how="inner"
    )

    print(f"按每天 cycle 数最多前 {top_n} 个 data_name 过滤后：")
    print("  不同日期数：", df_filtered["_date_for_limit"].nunique())
    print("  不同 data_name 数：", df_filtered["data_name"].nunique())
    print("  行数：", len(df_filtered))

    return df_filtered


def sample_user_by_day(df_user: pd.DataFrame,
                       max_per_day: int = 20,
                       seed: int = 42) -> pd.DataFrame:
    """
    对单个用户的数据按“日期”分组，每个(人, 日期)最多保留 max_per_day 条记录。

    注意：
    - 这里假设已经调用过 filter_top_data_names_per_day，
      因此 df_user 中已经有 "_date_for_limit" 列；
    - 如果没有，则在这里临时生成一个。
    """
    df_user = ensure_date_column(df_user, date_col="_date_for_limit")

    return (
        df_user.groupby(["user_id", "_date_for_limit"], group_keys=False)
               .apply(lambda g: g.sample(n=min(len(g), max_per_day), random_state=seed))
               .reset_index(drop=True)
    )


def load_one_user_file(path: str) -> pd.DataFrame:
    """
    读取单个用户的 Excel 特征文件，并把各个 sheet 合并成一张表。
    """
    df_prv = pd.read_excel(path, sheet_name="feature_prv")
    df_time = pd.read_excel(path, sheet_name="feature_time")
    df_welch1 = pd.read_excel(path, sheet_name="feature_welch_1")
    df_welch2 = pd.read_excel(path, sheet_name="feature_welch_2")
    df_ref = pd.read_excel(path, sheet_name="reference_time")
    df_freq1 = pd.read_excel(path, sheet_name="feature_frequency_1")
    df_freq2 = pd.read_excel(path, sheet_name="feature_frequency_2")

    def smart_merge(left, right):
        cand_keys = [
            "user_id", "data_name",
            "cycle_number", "select_number", "total_select_number"
        ]
        keys = [k for k in cand_keys if k in left.columns and k in right.columns]
        # 避免重复列
        drop_cols = [c for c in right.columns if c in left.columns and c not in keys]
        right2 = right.drop(columns=drop_cols)
        return left.merge(right2, on=keys, how="left")

    base = df_prv
    base = smart_merge(base, df_time)
    base = smart_merge(base, df_welch1)
    base = smart_merge(base, df_welch2)
    base = smart_merge(base, df_ref)
    base = smart_merge(base, df_freq1)
    base = smart_merge(base, df_freq2)

    return base


# ========= 主流程：只负责“过滤 + 按人/按类划分” =========

def prepare_and_save_splits(
    feature_dir: str = FEATURE_DIR,
    user_info_path: str = USER_INFO_PATH,
    train_csv_path: str = TRAIN_CSV_PATH,
    test_csv_path: str = TEST_CSV_PATH,
):
    # 读用户列表
    user_info = pd.read_csv(user_info_path)
    user_info["user_id"] = user_info["user_id"].astype(str)
    user_info["user_id"] = user_info["user_id"].str.lstrip("_")

    all_df = []

    # 收集所有特征文件
    feature_files = []
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.xlsx")))
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.csv")))

    for path in tqdm(feature_files, desc="正在加载用户特征数据"):
        name = os.path.basename(path)
        if name.startswith("~$"):
            # 排除 Excel 临时文件
            continue

        # 默认：文件名就是 user_id
        user_id_from_name = os.path.splitext(os.path.basename(path))[0]

        # 读特征
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".xlsx"):
            df = load_one_user_file(path)
        else:
            continue

        # 确保有 user_id 列
        if "user_id" in df.columns:
            df["user_id"] = df["user_id"].astype(str)
            user_id_in_file = str(df["user_id"].iloc[0])
            if user_id_in_file != user_id_from_name:
                print(f"⚠ 文件 {path} 文件名={user_id_from_name} 和内容里的 user_id={user_id_in_file} 不一致，以内容为准")
                user_id_from_name = user_id_in_file
        else:
            df["user_id"] = user_id_from_name

        # 在用户列表里查年龄 -> label
        row = user_info[user_info["user_id"] == user_id_from_name]
        if row.empty:
            print(f"⚠ 在用户列表中找不到 user_id={user_id_from_name}，跳过这个文件")
            continue

        age = int(row["年龄"].iloc[0])
        label = age_to_group(age)
        df["label"] = label

        # ========= 新增：先按“每天 cycle 数最多的前 8 个 data_name”过滤 =========
        df = filter_top_data_names_per_day(df, top_n=8)

        if df.empty:
            print(f"⚠ 用户 {user_id_from_name} 在该过滤条件下没有任何样本，跳过。")
            continue

        # ========= 然后再按天采样（每天最多 max_per_day 条） =========
        df = sample_user_by_day(df, max_per_day=MAX_PER_DAY_READ, seed=42)

        all_df.append(df)

    if not all_df:
        raise RuntimeError("没有成功读取到任何用户数据，请检查特征路径和用户列表。")

    # 合并所有用户
    df_all = pd.concat(all_df, ignore_index=True)

    print("全部数据形状：", df_all.shape)
    print("用户数量：", df_all["user_id"].nunique())

    # === 只做“按人 + 按类”划分，不做任何特征处理 ===
    train_df, test_df = split_by_person_stratified(
        df_all,
        user_id_col="user_id",
        label_col="label",
        train_person_ratio=0.8,
        train_per_class=10000,
        test_per_class=3000,
        seed=42,
    )

    # 删除采样临时列
    for col in ["_date_for_limit"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    # 原样保存（含所有原始列 + label）
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    print(f"训练集原始数据已保存到：{train_csv_path}")
    print(f"测试集原始数据已保存到：{test_csv_path}")


if __name__ == "__main__":
    prepare_and_save_splits()
