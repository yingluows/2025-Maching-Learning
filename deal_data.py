# deal_data.py
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from split import split_by_person_stratified

# ========= 路径配置（根据需要修改） =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"

FEATURE_DIR = os.path.join(BASE_DIR, "ppgfeature_v114")
USER_INFO_PATH = os.path.join(BASE_DIR, "用户疾病分类统计.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "Data_splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出“原始划分后的数据”（这里的“原始”指：未做 select_feature 的特征清洗，但已经做了“按天平均”）
TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "train_raw_v9.csv")
TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "test_raw_v9.csv")


# ========= 通用函数 =========

def age_to_group(age: int) -> int:
    """把年龄映射到年龄段 label（保留你原来的 0-5 分组逻辑）。"""
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
        df[date_col] = np.nan

    return df


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


def daily_mean_aggregate(df_user: pd.DataFrame, date_col: str = "_date_for_limit") -> pd.DataFrame:
    """
    核心策略：不随机抽取样本了 -> 改为每人每天对数值特征取平均。
    - 按 (user_id, date_col) 聚合
    - 数值列：mean
    - 非数值列：取第一条（仅用于保留必要字段；后续 select_feature 会删除非特征列）
    """
    df_user = ensure_date_column(df_user, date_col=date_col).copy()
    df_user = df_user.dropna(subset=[date_col])

    if df_user.empty:
        return df_user

    # 明确保留列
    must_keep_first = []
    for c in ["user_id", "label", date_col]:
        if c in df_user.columns:
            must_keep_first.append(c)

    # 识别数值列：除了 label 也可以平均，但 label 必须保持一致，所以从数值聚合中剔除 label
    numeric_cols = df_user.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["label"]]

    # 非数值列（object 等）——取第一条
    non_numeric_cols = [c for c in df_user.columns if c not in numeric_cols]

    # 构造聚合字典
    agg = {}
    for c in numeric_cols:
        agg[c] = "mean"
    for c in non_numeric_cols:
        # 对非数值列，取第一条；包括 user_id、label、date_col
        agg[c] = "first"

    df_day = (
        df_user
        .groupby(["user_id", date_col], as_index=False)
        .agg(agg)
    )

    # 强制 label 为 int（若原本是分组 label）
    if "label" in df_day.columns:
        df_day["label"] = pd.to_numeric(df_day["label"], errors="coerce").astype("Int64")

    return df_day


# ========= 主流程：过滤用户 + 每日平均 + 按人划分 =========

def prepare_and_save_splits(
    feature_dir: str = FEATURE_DIR,
    user_info_path: str = USER_INFO_PATH,
    train_csv_path: str = TRAIN_CSV_PATH,
    test_csv_path: str = TEST_CSV_PATH,
    train_person_ratio: float = 0.8,
    seed: int = 42,
):
    # 读用户列表
    user_info = pd.read_csv(user_info_path)
    user_info["user_id"] = user_info["user_id"].astype(str).str.lstrip("_")

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

        # 用户信息匹配
        row = user_info[user_info["user_id"] == str(user_id_from_name)]
        if row.empty:
            print(f"⚠ 在用户列表中找不到 user_id={user_id_from_name}，跳过")
            continue

        age = int(row["年龄"].iloc[0])
        label = age_to_group(age)
        df["label"] = label

        # ========= 50岁疾病过滤（按你的新要求） =========
        # <= 50：必须完全无疾病
        # > 50：必须有“循环系统疾病”
        has_any_disease = any(has_disease_value(row[col].iloc[0]) for col in disease_cols if col in row.columns)
        has_circulatory = has_disease_value(row["循环系统疾病"].iloc[0]) if "循环系统疾病" in row.columns else False

        #if age <= 50:
            #if has_any_disease:
                #continue
        #else:
            #if not has_circulatory:
                #continue

        # ========= 每人每天平均（替代随机抽取） =========
        df_day = daily_mean_aggregate(df, date_col="_date_for_limit")
        if df_day.empty:
            continue

        all_df.append(df_day)

    if not all_df:
        raise RuntimeError("没有成功读取到任何用户数据：请检查 FEATURE_DIR / USER_INFO_PATH / 过滤条件。")

    df_all = pd.concat(all_df, ignore_index=True)

    print("全部数据形状（按人-天聚合后）：", df_all.shape)
    print("用户数量：", df_all["user_id"].nunique())
    print("天数样本数：", len(df_all))

    # ========= 按“人”划分训练/测试（人不交叉），不再做按类抽样 =========
    train_df, test_df = split_by_person_stratified(
        df_all,
        user_id_col="user_id",
        label_col="label",
        train_person_ratio=train_person_ratio,
        seed=seed,
        # 关键：不抽样
        do_stratified_sample=False,
    )

    # 这里先保留 _date_for_limit，方便你排查；后续 select_feature 会删除非特征列
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    print(f"训练集已保存到：{train_csv_path}")
    print(f"测试集已保存到：{test_csv_path}")


if __name__ == "__main__":
    prepare_and_save_splits()
