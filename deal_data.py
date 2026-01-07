# deal_data.py
import os
import glob
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from split import split_by_person_stratified


# ======================
# 默认路径：以当前脚本所在目录为项目根目录
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FEATURE_DIR = os.path.join(BASE_DIR, "ppgfeature_v114")
DEFAULT_USER_INFO_PATH = os.path.join(BASE_DIR, "用户疾病分类统计.csv")

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

DEFAULT_TRAIN_CSV_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "train_raw_v5.csv")
DEFAULT_TEST_CSV_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "test_raw_v5.csv")

# 每人每天最多保留的样本数（你原来的逻辑保留；如果不想限制，可设为 None）
DEFAULT_MAX_PER_DAY_READ = None


# ========= 通用函数 =========

def age_to_group(age: int) -> int:
    """把年龄映射到年龄段 label（你原来的分段保持不变）"""
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
      - 按日期 + data_name 统计 cycle 数量（行数）
      - 每天选 cycle 数最多的前 top_n 个 data_name
      - 只保留这些 (日期, data_name) 对应的所有行
    """
    if "data_name" not in df.columns or "cycle_number" not in df.columns:
        print("⚠ 警告：数据中不存在 'data_name' 或 'cycle_number' 列，无法按每天前 N 个 data_name 过滤，原样返回。")
        return df

    df = ensure_date_column(df, date_col="_date_for_limit")

    group = (
        df.groupby(["_date_for_limit", "data_name"])["cycle_number"]
        .count()
        .reset_index(name="cycle_count")
    )

    group_sorted = group.sort_values(["_date_for_limit", "cycle_count"], ascending=[True, False])
    top = group_sorted.groupby("_date_for_limit").head(top_n)

    df_filtered = df.merge(
        top[["_date_for_limit", "data_name"]],
        on=["_date_for_limit", "data_name"],
        how="inner"
    )

    return df_filtered


def sample_user_by_day(df_user: pd.DataFrame, max_per_day: int, seed: int = 42) -> pd.DataFrame:
    """
    对单个用户的数据按“日期”分组，每个(人, 日期)最多保留 max_per_day 条记录。
    如果 max_per_day 为 None，则不采样直接返回。
    """
    if max_per_day is None:
        return df_user.copy()

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
        cand_keys = ["user_id", "data_name", "cycle_number", "select_number", "total_select_number"]
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
    """判断一个疾病单元格是否表示“有疾病”"""
    if pd.isna(x):
        return False
    x = str(x).strip()
    if x in ("0", "无", "否", "", "nan", "NaN"):
        return False
    return True


def aggregate_by_dataname_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 关键：把每个 (user_id, data_name) 压成 1 条样本（均值特征）
    - 数值列：mean
    - label：first（同一 user 应一致）
    """
    df = df.copy()

    if "user_id" not in df.columns or "data_name" not in df.columns:
        raise ValueError("聚合需要 'user_id' 和 'data_name' 列。")

    if "label" not in df.columns:
        raise ValueError("聚合需要 'label' 列。")

    group_cols = ["user_id", "data_name"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # label 不能求 mean
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    agg_dict = {c: "mean" for c in numeric_cols}
    agg_dict["label"] = "first"

    out = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # 一些统计信息
    print("[聚合] 每个 (user_id, data_name) 变成 1 条样本后：", out.shape)
    print("[聚合] 样本数(=data_name数):", len(out), "用户数:", out["user_id"].nunique())
    print("[聚合] 各类样本数:\n", out["label"].value_counts().sort_index())

    return out


# ========= 主流程 =========

def prepare_and_save_splits(
    feature_dir: str,
    user_info_path: str,
    out_dir: str,
    max_per_day_read: int | None,
    filter_top_n: int | None,
    train_person_ratio: float,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)
    train_csv_path = os.path.join(out_dir, "train_raw_v5.csv")
    test_csv_path = os.path.join(out_dir, "test_raw_v5.csv")

    # 读用户列表
    user_info = pd.read_csv(user_info_path)
    user_info["user_id"] = user_info["user_id"].astype(str).str.lstrip("_")

    all_df = []

    # 收集所有特征文件
    feature_files = []
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.xlsx")))
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.csv")))

    if not feature_files:
        raise RuntimeError(f"在 {feature_dir} 下没有找到任何 .xlsx/.csv 特征文件")

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

        # user_id 对齐
        if "user_id" in df.columns:
            df["user_id"] = df["user_id"].astype(str)
            user_id_in_file = str(df["user_id"].iloc[0])
            if user_id_in_file != user_id_from_name:
                user_id_from_name = user_id_in_file
        else:
            df["user_id"] = user_id_from_name

        row = user_info[user_info["user_id"] == user_id_from_name]
        if row.empty:
            print(f"⚠ 在用户列表中找不到 user_id={user_id_from_name}，跳过")
            continue

        age = int(row["年龄"].iloc[0])
        label = age_to_group(age)
        df["label"] = label

        # 你原来的“年龄×疾病”筛选逻辑保留
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

        if age <= 50:
            if has_any_disease:
                continue
        else:
            if not has_circulatory:
                continue

        # 可选：每天选 top_n data_name（如果你想完全不做这一步，传 filter_top_n=None）
        if filter_top_n is not None:
            df = filter_top_data_names_per_day(df, top_n=filter_top_n)

        if df.empty:
            continue

        # 可选：按天限流（如果你想完全不限制，传 max_per_day_read=None）
        df = sample_user_by_day(df, max_per_day=max_per_day_read, seed=seed)

        all_df.append(df)

    if not all_df:
        raise RuntimeError("没有成功读取到任何用户数据，请检查路径、筛选条件。")

    df_all = pd.concat(all_df, ignore_index=True)
    print("全部数据形状：", df_all.shape, "用户数量：", df_all["user_id"].nunique())

    # 1) 先按人划分（不抽样：train_per_class=None）
    train_df, test_df = split_by_person_stratified(
        df_all,
        user_id_col="user_id",
        label_col="label",
        train_person_ratio=train_person_ratio,
        train_per_class=None,      # ✅ 关键：不在这里截断/抽样
        test_per_class=None,
        seed=seed,
    )

    # 2) 再分别对 train/test 做均值聚合（每个 data_name 变 1 条）
    #    ✅ 关键：测试集也必须聚合，样本定义一致
    # 先去掉临时列（如果存在）
    for col in ["_date_for_limit"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    train_df = aggregate_by_dataname_mean(train_df)
    test_df = aggregate_by_dataname_mean(test_df)

    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ 训练集已保存：{train_csv_path}")
    print(f"✅ 测试集已保存：{test_csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", type=str, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--user_info", type=str, default=DEFAULT_USER_INFO_PATH)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_per_day", type=int, default=DEFAULT_MAX_PER_DAY_READ, help="每人每天最多保留多少条；设为 -1 表示不限制")
    p.add_argument("--top_n", type=int, default=5, help="每天 cycle 数最多的前 N 个 data_name；设为 -1 表示不启用")
    p.add_argument("--train_ratio", type=float, default=0.7, help="按人划分训练集比例")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_per_day = None if args.max_per_day == -1 else args.max_per_day
    top_n = None if args.top_n == -1 else args.top_n

    prepare_and_save_splits(
        feature_dir=args.feature_dir,
        user_info_path=args.user_info,
        out_dir=args.out_dir,
        max_per_day_read=max_per_day,
        filter_top_n=top_n,
        train_person_ratio=args.train_ratio,
        seed=args.seed,
    )
