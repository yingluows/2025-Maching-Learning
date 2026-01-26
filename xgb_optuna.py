import matplotlib
import pandas as pd
import numpy as np
import os
import optuna
import ast
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from xgboost import XGBClassifier

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# ———— 公共工具函数 ————

# 这些列都是 "[a, b]" 格式的字符串，需要拆成两列
BRACKET_FEATURE_COLS = [
    # feature_time 里的 FDA/FDB
    "FDA1","FDB1",

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

"""
# 构造新特征
def generate_new_features(df: pd.DataFrame) -> pd.DataFrame:
    ancient_data = df.drop(columns=["feature_name"], errors='ignore')
    data = ancient_data.select_dtypes(include=[np.number])
    new_features = pd.DataFrame(index=data.index)

    new_features["row_std"] = data.std(axis=1)
    new_features["row_min"] = data.min(axis=1)
    new_features["row_max"] = data.max(axis=1)

    if "FDA1x" in data.columns and "FDA1y" in data.columns:
        new_features["fda1_slope"] = data["FDA1y"] / (data["FDA1x"] + 1e-6)
    if "FDB1x" in data.columns and "FDB1y" in data.columns:
        new_features["fdb1_slope"] = data["FDB1y"] / (data["FDB1x"] + 1e-6)

    if set(["HR", "PRT"]).issubset(data.columns):
        new_features["delta_hr_prt"] = data["HR"] - data["PRT"]
        new_features["hr_prt_ratio"] = data["HR"] / (data["PRT"] + 1e-6)
        new_features["log_hr_diff"] = np.log1p(np.abs(data["HR"] - data["PRT"]))

    if set(["VF1", "VF2", "VF3", "VF4", "VF5"]).issubset(data.columns):
        new_features["sum_vf"] = data[["VF1", "VF2", "VF3", "VF4", "VF5"]].sum(axis=1)
        for k in ["VF1", "VF2", "VF3", "VF4", "VF5"]:
            new_features[k.lower() + "_ratio"] = data[k] / (new_features["sum_vf"] + 1e-6)

    # c*/s* 的比例特征
    if set(["c0", "c1", "c2", "c3", "c4", "c5"]).issubset(data.columns):
        new_features["sum_c"] = data[[f"c{i}" for i in range(6)]].sum(axis=1)
        for i in range(6):
            new_features[f"c{i}_ratio"] = data[f"c{i}"] / (new_features["sum_c"] + 1e-6)

    if set(["s0", "s1", "s2", "s3", "s4", "s5"]).issubset(data.columns):
        new_features["sum_s"] = data[[f"s{i}" for i in range(6)]].sum(axis=1)
        for i in range(6):
            new_features[f"s{i}_ratio"] = data[f"s{i}"] / (new_features["sum_s"] + 1e-6)

    return pd.concat([data, new_features], axis=1)

"""    

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
    df_prv  = pd.read_excel(path, sheet_name="feature_prv")
    df_time = pd.read_excel(path, sheet_name="feature_time")
    df_welch = pd.read_excel(path, sheet_name="feature_welch")
    df_ref  = pd.read_excel(path, sheet_name="reference_time")
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

    # 把 "[a,b]" 列拆成 col_x/col_y
    # base = expand_bracket_features(base)

    return base



# ———— 数据拆分策略：split_by_person_random ————
from sklearn.utils import resample

def split_by_person_random(df: pd.DataFrame,
                           train_sample_limit: int = 1000,
                           test_sample_limit: int = 300,
                           max_per_person: int = 30,
                           train_ratio: float = 0.8,
                           seed: int = 42,
                           verbose: bool = True):
    """
    针对新数据：
    - 一行 = 一次测量
    - user_id = 用户ID
    - 按 user_id 划分训练 / 测试，不同用户不会交叉
    """
    np.random.seed(seed)
    df = df.copy()

    if "user_id" not in df.columns:
        raise ValueError("数据里没有 user_id 列，无法按人划分！")

    # 用user_id 当 group_id
    df["group_id"] = df["user_id"].astype(str)

    # 用“每个用户”为一个 group
    df["person_day"] = df["group_id"]

    # 按人划分 train / test
    unique_people = df["group_id"].unique()
    np.random.shuffle(unique_people)
    split_idx = int(len(unique_people) * train_ratio)
    train_people = unique_people[:split_idx]
    test_people  = unique_people[split_idx:]

    train_raw = df[df["group_id"].isin(train_people)]
    test_raw  = df[df["group_id"].isin(test_people)]

    # 每人最多保留 max_per_person 条样本
    def limit_by_person(df_raw):
        return df_raw.groupby("person_day", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_per_person), random_state=seed)
        ).reset_index(drop=True)

    train_df = limit_by_person(train_raw)
    test_df  = limit_by_person(test_raw)

    # 控制总样本量
    def limit_sample_count(df_raw, max_samples):
        if len(df_raw) <= max_samples:
            return df_raw
        else:
            return df_raw.sample(n=max_samples, random_state=seed)

    train_df = limit_sample_count(train_df, train_sample_limit)
    test_df  = limit_sample_count(test_df,  test_sample_limit)

    if verbose:
        print(f"训练集人数: {train_df['group_id'].nunique()}, 样本数: {len(train_df)}")
        print(f"测试集人数: {test_df['group_id'].nunique()}, 样本数: {len(test_df)}")

    return train_df, test_df

def split_by_person_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    max_per_person: int = 30,
    train_per_class: int = 1000,
    test_per_class: int = 300,
    seed: int = 42,
):
    """
    按人划分 + 每人最多 max_per_person 条 + 每类固定样本数（1000 / 300）
    1. 按 user_id 随机划分 train/test 人群
    2. 各自内部，每人最多保留 max_per_person 条记录
    3. 然后在 train/test 内部分别按 label 分层抽样：
        - 训练集：每个 label 最多 train_per_class 条
        - 测试集：每个 label 最多 test_per_class 条
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    # === 第一步：按人划分 train/test ===
    persons = df[user_id_col].dropna().unique().tolist()
    rng.shuffle(persons)
    n = len(persons)
    cut = int(round(n * train_person_ratio))
    cut = max(min(cut, n - 1), 1)   # 确保 train/test 都不为空

    train_ids = set(persons[:cut])
    test_ids  = set(persons[cut:])

    train_raw = df[df[user_id_col].isin(train_ids)].copy()
    test_raw  = df[df[user_id_col].isin(test_ids)].copy()

    # === 第二步：每人最多 max_per_person 条 ===
    def limit_per_person(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        return (
            d.groupby(user_id_col, group_keys=False)
             .apply(lambda g: g.sample(n=min(len(g), max_per_person), random_state=seed))
             .reset_index(drop=True)
        )

    train_limited = limit_per_person(train_raw)
    test_limited  = limit_per_person(test_raw)

    # === 第三步：按 label 分层抽样 ===
    def stratified_sample(d: pd.DataFrame, per_class: int) -> pd.DataFrame:
        parts = []
        for label, g in d.groupby(label_col):
            n_available = len(g)
            if n_available == 0:
                continue
            n_take = min(per_class, n_available)
            parts.append(g.sample(n=n_take, random_state=seed))
        if parts:
            return pd.concat(parts, ignore_index=True)
        else:
            return d.iloc[0:0].copy()

    train_final = stratified_sample(train_limited, train_per_class)
    test_final  = stratified_sample(test_limited, test_per_class)

    print("=== 划分结果统计 ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final


# ———— 主流程 ————
import glob

MAX_ROWS_PER_USER = 50

# === 路径配置 ===
feature_dir = r"D:/2025_Stage/Code/XGB/ppgfeature"   # 特征文件夹
user_info_path = r"D:/2025_Stage/Code/XGB/用户列表.csv"  # 里面有“年龄”那一列

# 年龄 -> 年龄段标签
def age_to_group(age: int) -> int:
    if age < 20:
        return 0
    elif age < 30:
        return 1
    elif age < 40:
        return 2
    elif age < 50:
        return 3
    elif age < 60:
        return 4
    else:
        return 5

# 读用户列表
user_info = pd.read_csv(user_info_path)
user_info["user_id"] = user_info["user_id"].astype(str)  # 确保是字符串
# 去掉 user_id 前缀的 "_"，保证与文件名一致
user_info["user_id"] = user_info["user_id"].str.lstrip("_")


all_df = []

# 同时支持 csv / xlsx 两种
feature_files = []
for f in glob.glob(os.path.join(feature_dir, "*.xlsx")) + \
                glob.glob(os.path.join(feature_dir, "*.csv")):
    name = os.path.basename(f)
    if name.startswith("~$"):
        continue  # 排除 Excel 临时文件
    feature_files.append(f)

for path in tqdm(feature_files, desc="正在加载用户特征数据"):
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
    if len(df) > MAX_ROWS_PER_USER:
        df = df.sample(n=MAX_ROWS_PER_USER, random_state=42)

    all_df.append(df)

# 合并所有用户
df_all = pd.concat(all_df, ignore_index=True)
df_all = expand_bracket_features(df_all)
print("全部数据形状：", df_all.shape)
print("用户数量：", df_all["user_id"].nunique())


# === 按人拆分训练/测试 ===
train_df, test_df = split_by_person_stratified(
    df_all,
    user_id_col="user_id",
    label_col="label",
    train_person_ratio=0.8,  # 按人 8:2
    max_per_person=30,
    train_per_class=1000,
    test_per_class=300,
    seed=42,
)



# === 特征清理 ===
train_df, test_df = deal_file(train_df, test_df)

# 拆分特征与标签
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
X_test  = test_df.drop(columns=["label"])
y_test  = test_df["label"]


# 训练前清理无穷值为 NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

# Optuna 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
    }

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        **params
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("xgb", xgb)
    ])

    return cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("最优参数：", study.best_params)
print("最优交叉验证准确率：", study.best_value)

# 训练与评估 —— 使用最优参数的 XGBClassifier
best_params = study.best_params.copy()
xgb_best = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
    **best_params
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("xgb", xgb_best)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


report_dict = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).T
df_report.to_csv("D:/2025_Stage/Code/XGB/Save_fig/classification_report.csv", encoding="utf-8-sig")
print("分类报告已保存到: classification_report.csv")

# 混淆矩阵可视化
ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test,
                                      display_labels=sorted(y_train.unique()), cmap=plt.cm.Oranges)
plt.title("XGB Confusion matrix")
plt.savefig("D:\\2025_Stage\\Code\\XGB\\Save_fig\\xgb_confusion_matrix_optuna.png", dpi=300)
plt.tight_layout()
plt.show()

# 保存模型
joblib.dump(pipeline, "D:/2025_Stage/Code/XGB/Save_model/xgb_model.pkl")
print("模型已保存到 xgb_model.pkl")

# === SHAP 模型解释 ===
import shap

# 取一部分样本，避免计算太慢
sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)

# 创建通用型 SHAP 解释器（对 pipeline 直接做预测）
explainer = shap.Explainer(pipeline.predict, sample_X)
shap_values = explainer(sample_X)

# 绘制 SHAP Summary Dot 图（单特征分布）
plt.figure()
shap.summary_plot(shap_values, sample_X, show=False)
plt.title("XGB SHAP Summary (Dot)")
plt.savefig("D:\\2025_Stage\\Code\\XGB\\Save_fig\\xgb_shap_summary_dot.png", dpi=300, bbox_inches='tight')
plt.close()

# 绘制 SHAP Bar Summary 图（平均重要性）
plt.figure()
shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
plt.title("XGB SHAP Summary (Bar)")
plt.savefig("D:\\2025_Stage\\Code\\XGB\\Save_fig\\xgb_shap_summary_bar.png", dpi=300, bbox_inches='tight')
plt.close()

print("SHAP 解释图已保存：xgb_shap_summary_dot.png 与 xgb_shap_summary_bar.png")
