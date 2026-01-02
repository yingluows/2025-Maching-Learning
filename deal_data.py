# deal_data.py
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from split import split_by_person_stratified, split_by_person_day_window_stratified


# ========= 路径配置（根据需要修改） =========
FEATURE_DIR = r"D:/2025_Stage/Code/XGB/ppgfeature_v114"
USER_INFO_PATH = r"D:/2025_Stage/Code/XGB/用户疾病分类统计.csv"

OUTPUT_DIR = r"D:/2025_Stage/Code/XGB/Data_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出的是“原始划分后的数据”，不做特征处理
TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "train_raw_v7.csv")
TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "test_raw_v7.csv")


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


def filter_top_data_names_per_day(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    对单个用户的数据：
      - 先按日期 + data_name 统计 cycle 数量（行数）
      - 对每个“日期”，选出 cycle 数最多的前 top_n 个 data_name
      - 只保留这些 (日期, data_name) 对应的所有行
      - 如果某天 data_name 少于 top_n，则全保留

    注意：这里“不随机、不限量”，保留被选中 data_name 的全部记录。
    """
    if "data_name" not in df.columns or "cycle_number" not in df.columns:
        print("⚠ 警告：数据中不存在 'data_name' 或 'cycle_number' 列，无法按每天前 N 个 data_name 过滤，原样返回。")
        return df

    df = ensure_date_column(df, date_col="_date_for_limit")

    # 统计每个 (日期, data_name) 的 cycle 数（用行数即可）
    group = (
        df.groupby(["_date_for_limit", "data_name"])["cycle_number"]
          .count()
          .reset_index(name="cycle_count")
    )

    # 每天取 cycle_count 最大的前 top_n 个 data_name
    group_sorted = group.sort_values(
        ["_date_for_limit", "cycle_count"],
        ascending=[True, False]
    )
    top = group_sorted.groupby("_date_for_limit").head(top_n)

    # 只保留这些 (日期, data_name) 的行（保留“所有数据”）
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
            continue

        user_id_from_name = os.path.splitext(os.path.basename(path))[0]

        # 读特征
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".xlsx"):
            df = load_one_user_file(path)
        else:
            continue

        # 确保 user_id
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
        df["age"] = age

        # ========= 按“年龄 × 疾病”筛选用户 =========
        disease_cols = [
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

        has_any_disease = any(has_disease_value(row[col].iloc[0]) for col in disease_cols)
        has_circulatory = has_disease_value(row["循环系统疾病"].iloc[0])

        # 1) <=50：必须完全无疾病
        if age <= 50:
            if has_any_disease:
                continue
        # 2) >50：必须有循环系统疾病
        else:
            if not has_circulatory:
                continue

        # ========= 核心修改：每人每天取 cycle_count 最大的前 1 个 data_name，保留其所有数据 =========
        df = filter_top_data_names_per_day(df, top_n=1)

        if df.empty:
            print(f"⚠ 用户 {user_id_from_name} 在该过滤条件下没有任何样本，跳过。")
            continue

        # ✅ 不再做按天随机采样，直接保留
        all_df.append(df)

    if not all_df:
        raise RuntimeError("没有成功读取到任何用户数据，请检查特征路径和用户列表。")

    # 合并所有用户
    df_all = pd.concat(all_df, ignore_index=True)

    print("全部数据形状：", df_all.shape)
    print("用户数量：", df_all["user_id"].nunique())

    # === 只做“按人 + 按类”划分（0.8 人进训练，0.2 人进测试，不改） ===
    train_df, test_df = split_by_person_stratified(
        df_all,
        user_id_col="user_id",
        label_col="label",
        train_person_ratio=0.8,   # ✅ 按你的要求保持不变
        train_per_class=10000,
        test_per_class=2000,
        seed=42,
    )

    # 删除临时日期列（如果存在）
    for col in ["_date_for_limit"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    # 原样保存
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    print(f"训练集原始数据已保存到：{train_csv_path}")
    print(f"测试集原始数据已保存到：{test_csv_path}")


if __name__ == "__main__":
    prepare_and_save_splits()
