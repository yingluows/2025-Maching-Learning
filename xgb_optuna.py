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

def deal_file(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    针对新的 5-sheet 特征表：
    - 去掉所有不参与训练的标识列
    - 删除所有 "[a,b]" 的原始字符串列（FDA1、frequency_lobe*、ppg_peak...）
    - 进一步按列名规则删掉冗余特征（Welch、峰谷坐标等）
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

    # 额外识别列（你在那段 drop_cols 里写的那些）
    id_like_cols = ["user_id", "data_name", "cycle_number",
                    "select_number", "total_select_number", "_date_for_limit"]

    def _process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 0. 必须先保证有 label
        if "label" not in df.columns:
            raise ValueError("数据中没有 label 列，请确认在调用 deal_file 前已经打好标签。")

        # 1. 删掉各种非特征列
        df = df.drop(
            columns=[c for c in drop_not_features if c in df.columns],
            errors="ignore"
        )

        # 2. 删掉所有 "[a,b]" 的原始字符串列（BRACKET_FEATURE_COLS 你之前已经定义了）
        df = df.drop(
            columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns],
            errors="ignore"
        )

        # ====== drop_cols ======
        label = df["label"]
        feature_df = df.drop(columns=["label"])

        drop_cols = []

        # 2.1 删除 Welch 的所有列（spectrum*, frequency_lobe*）
        #drop_cols += [c for c in feature_df.columns
                      #if c.startswith("spectrum") or c.startswith("frequency_lobe")]

        # 2.2 删除坐标类峰谷信息（peak/valley 字样的列）
        #drop_cols += [c for c in feature_df.columns
                      #if "peak" in c.lower() or "valley" in c.lower()]

        # 2.3 删除识别列/日期辅助列
        drop_cols += [c for c in id_like_cols if c in feature_df.columns]

        # 2.4 删除仍为字符串/对象类型的列（label 已经单独拿出来了）
        drop_cols += [c for c in feature_df.columns
                      if feature_df[c].dtype == "object"]

        # 真正删除
        feature_df = feature_df.drop(columns=list(set(drop_cols)), errors="ignore")

        # 2.5 再保险：只保留数值型特征列（防止还有残留的非数值）
        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"处理后保留特征数：{feature_df.shape[1]}")

        # 拼回 label
        df_processed = pd.concat([feature_df, label], axis=1)

        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)

    return train_df, test_df


def sample_user_by_day(df_user: pd.DataFrame,
                       max_per_day: int = 20,
                       seed: int = 42) -> pd.DataFrame:
    """
    对单个用户的数据按“日期”分组，每个(人, 日期)最多保留 max_per_day 条记录。
    日期从 data_name 末尾的 YYYYMMDDhhmmss 中解析出来。
    """
    df_user = df_user.copy()

    if "data_name" in df_user.columns:
        # 提取 data_name 里的 YYYYMMDD，形如 ..._20240603013426
        ts_str = df_user["data_name"].astype(str).str.extract(r"_(\d{8})\d{6}$", expand=False)
        dates = pd.to_datetime(ts_str, format="%Y%m%d", errors="coerce")
        df_user["_date_for_limit"] = dates.dt.strftime("%Y-%m-%d")
    else:
        # 没有 data_name，就只能当成一个“虚拟日期”
        df_user["_date_for_limit"] = "NA"

    # 按 (user_id, 日期) 分组，每组最多 max_per_day 条
    return (
        df_user.groupby(["user_id", "_date_for_limit"], group_keys=False)
               .apply(lambda g: g.sample(n=min(len(g), max_per_day), random_state=seed))
               .reset_index(drop=True)
    )


def load_one_user_file(path: str) -> pd.DataFrame:
    df_prv  = pd.read_excel(path, sheet_name="feature_prv")
    df_time = pd.read_excel(path, sheet_name="feature_time")
    df_welch1 = pd.read_excel(path, sheet_name="feature_welch_1")
    df_welch2 = pd.read_excel(path, sheet_name="feature_welch_2")
    df_ref  = pd.read_excel(path, sheet_name="reference_time")
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

    # 把 "[a,b]" 列拆成 col_x/col_y
    # base = expand_bracket_features(base)

    return base

def split_by_person_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    train_per_class: int = 1000,
    test_per_class: int = 300,
    seed: int = 42,
):
    """
    版本2：假设“按天抽样”已经在读取阶段完成，这里只负责：
      1. 按人划分 train/test（人不交叉）
      2. 在各自内部按 label 分层抽样：train 每类≤train_per_class，test 每类≤test_per_class
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

    # === 第二步：按 label 分层抽样（不再按天/按人限一次） ===
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

    train_final = stratified_sample(train_raw, train_per_class)
    test_final  = stratified_sample(test_raw, test_per_class)

    print("=== 划分结果统计（按人 + 按类） ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final



# ———— 主流程 ————
import glob

# === 路径配置 ===
feature_dir = r"D:/2025_Stage/Code/XGB/ppgfeature_v114"   # 特征文件夹
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

MAX_PER_DAY_READ = 5

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
    df = sample_user_by_day(df, max_per_day=MAX_PER_DAY_READ, seed=42)

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






