# deal_data.py
import os
import glob
import ast
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from split import split_by_person_stratified

# ========= 路径配置（根据需要修改） =========
FEATURE_DIR = r"D:/2025_Stage/Code/XGB/ppgfeature_v114"
USER_INFO_PATH = r"D:/2025_Stage/Code/XGB/用户列表.csv"

OUTPUT_DIR = r"D:/2025_Stage/Code/XGB/Data_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "test.csv")

MAX_PER_DAY_READ = 5  # 每人每天最多读取的样本数


# ========= 公共配置 =========

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


def age_to_group(age: int) -> int:
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


# ========= 处理 “[a,b]” 字符串列 =========

def safe_eval(val):
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
    原始 col 列保留不动（之后在 deal_file 中统一删）
    """
    df = df.copy()
    for col in BRACKET_FEATURE_COLS:
        if col in df.columns:
            two_cols = df[col].apply(safe_eval).apply(pd.Series)
            df[f"{col}_x"] = two_cols[0]
            df[f"{col}_y"] = two_cols[1]
    return df


def deal_file(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - 去掉各种非特征列
    - 使用 BRACKET_FEATURE_COLS 的 *_x, *_y 构造一个新特征 col_mag = sqrt(x^2 + y^2)
    - 然后删掉原始字符串列 + *_x, *_y
    - 最终只保留数值特征 + label
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

        # 1. 删除非特征列
        df = df.drop(
            columns=[c for c in drop_not_features if c in df.columns],
            errors="ignore"
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
            errors="ignore"
        )

        # 4. 拆分 label / feature_df
        label = df["label"]
        feature_df = df.drop(columns=["label"])

        # 5. 删掉所有 *_x, *_y，只保留 *_mag
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")],
            errors="ignore"
        )

        # 6. 删除所有 object 列
        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"[deal_file] 处理后保留特征数：{feature_df.shape[1]}")

        df_processed = pd.concat([feature_df, label], axis=1)
        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)

    return train_df, test_df


# ========= 辅助函数 =========

def sample_user_by_day(df_user: pd.DataFrame,
                       max_per_day: int = 20,
                       seed: int = 42) -> pd.DataFrame:
    """
    对单个用户的数据按“日期”分组，每个(人, 日期)最多保留 max_per_day 条记录。
    日期从 data_name 末尾的 YYYYMMDDhhmmss 中解析出来。
    """
    df_user = df_user.copy()

    if "data_name" in df_user.columns:
        ts_str = df_user["data_name"].astype(str).str.extract(r"_(\d{8})\d{6}$", expand=False)
        dates = pd.to_datetime(ts_str, format="%Y%m%d", errors="coerce")
        df_user["_date_for_limit"] = dates.dt.strftime("%Y-%m-%d")
    else:
        df_user["_date_for_limit"] = "NA"

    return (
        df_user.groupby(["user_id", "_date_for_limit"], group_keys=False)
               .apply(lambda g: g.sample(n=min(len(g), max_per_day), random_state=seed))
               .reset_index(drop=True)
    )


def load_one_user_file(path: str) -> pd.DataFrame:
    """
    从单个用户的 Excel 文件中读取各 sheet，并合并。
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


# ========= 主流程：生成 train.csv / test.csv =========

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

    feature_files = []
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.xlsx")))
    feature_files.extend(glob.glob(os.path.join(feature_dir, "*.csv")))

    for path in tqdm(feature_files, desc="正在加载用户特征数据"):
        name = os.path.basename(path)
        if name.startswith("~$"):
            continue  # 排除 Excel 临时文件

        # 文件名就是 user_id
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

        # 在用户列表里查年龄
        row = user_info[user_info["user_id"] == user_id_from_name]
        if row.empty:
            print(f"⚠ 在用户列表中找不到 user_id={user_id_from_name}，跳过这个文件")
            continue

        age = int(row["年龄"].iloc[0])
        label = age_to_group(age)

        df["label"] = label
        df = sample_user_by_day(df, max_per_day=MAX_PER_DAY_READ, seed=42)

        all_df.append(df)

    if not all_df:
        raise RuntimeError("没有成功读取到任何用户数据，请检查特征路径和用户列表。")

    # 合并所有用户
    df_all = pd.concat(all_df, ignore_index=True)
    df_all = expand_bracket_features(df_all)

    print("全部数据形状：", df_all.shape)
    print("用户数量：", df_all["user_id"].nunique())

    # === 按人 / 按类拆分训练/测试 ===
    train_df, test_df = split_by_person_stratified(
        df_all,
        user_id_col="user_id",
        label_col="label",
        train_person_ratio=0.8,
        train_per_class=1500,
        test_per_class=300,
        seed=42,
    )

    # === 特征清理 ===
    train_df, test_df = deal_file(train_df, test_df)

    # 保存为 CSV
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    print(f"训练集已保存到：{train_csv_path}")
    print(f"测试集已保存到：{test_csv_path}")


if __name__ == "__main__":
    prepare_and_save_splits()
