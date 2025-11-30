import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import os
import optuna
import ast
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 数据文件路径和对应标签
file_list = [
    (["D:/2025_Stage/Code/Feature3/female_less20_feature.csv",
      "D:/2025_Stage/Code/Feature3/male_less20_feature.csv",
      "D:/2025_Stage/Code/Feature3/filed_less20_feature.csv"],0),
    (["D:/2025_Stage/Code/Feature3/female_20to29_feature.csv",
      "D:/2025_Stage/Code/Feature3/unfiled_20to29_feature.csv"],1),
    ("D:/2025_Stage/Code/Feature3/female_30to39_feature.csv",2),
    ("D:/2025_Stage/Code/Feature3/female_40to49_feature.csv",3),
    ("D:/2025_Stage/Code/Feature3/female_50to59_feature.csv",4),
    (["D:/2025_Stage/Code/Feature3/female_60to69_feature.csv",
      "D:/2025_Stage/Code/Feature3/female_ge70_feature.csv"],5)
]

# ———— 公共工具函数 ————

def safe_eval(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        else:
            return [None, None]
    except Exception:
        return [None, None]

def ensure_group_id(df: pd.DataFrame):
    """确保有 group_id 列；对 None 安全。"""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if 'group_id' not in df.columns:
        if 'feature_name' in df.columns:
            df = df.copy()
            df['group_id'] = df['feature_name'].astype(str).str.extract(r'^(\d+)', expand=False)
        else:
            df['group_id'] = pd.NA
    return df

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
        new_features["sum_vf"] = data[["VF1","VF2","VF3","VF4","VF5"]].sum(axis=1)
        for k in ["VF1","VF2","VF3","VF4","VF5"]:
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

# 定义冗余特征
redundant_cols  = ['feature_name', 'FDA1', 'AUI', 'FDB1']
redundant_cols2 = ['VF1','VF2','VF3','VF4','VF5',
                   'c0','c1','c2','c3','c4','c5',
                   's0','s1','s2','s3','s4','s5','HR']
redundant_cols3 = ['feature_name','FDA1','FDB1','AUI','LASI','DAR','AGI',
                   'SDPCR','SDVCR','FDPCR','SDVC','SDPC','FDPC','FDVC','FDVCR']

def deal_file(train_df: pd.DataFrame, test_df: pd.DataFrame, label: int):
    # 先丢弃冗余三类特征
    train_df.drop(columns=[col for col in redundant_cols3], inplace=True, errors='ignore')
    test_df.drop(columns=[col for col in redundant_cols3], inplace=True, errors='ignore')

    # 生成新特征
    train_df = generate_new_features(train_df)
    test_df = generate_new_features(test_df)

    # 再删除原始的 VF/c/s/HR 等冗余列
    train_df.drop(columns=[col for col in redundant_cols2], inplace=True, errors='ignore')
    test_df.drop(columns=[col for col in redundant_cols2], inplace=True, errors='ignore')

    train_df["label"] = label
    test_df["label"] = label
    return train_df, test_df

# ———— 数据拆分策略：split_by_person_random ————
from sklearn.utils import resample

def split_by_person_random(df: pd.DataFrame,
                           train_sample_limit: int = 1000,
                           test_sample_limit: int = 300,
                           max_per_day: int = 20,
                           train_ratio: float = 0.8,
                           seed: int = 42,
                           verbose: bool = True):
    np.random.seed(seed)

    # 字段提取
    df['group_id'] = df['feature_name'].astype(str).str.extract(r'^(\d+)', expand=False)
    df['date_human'] = df['feature_name'].astype(str).str.extract(r'^\d+_(\d+_\d+_\d+)', expand=False)
    df['date_compact'] = df['feature_name'].astype(str).str.extract(r'_(\d{8})\d{6}_\d+$', expand=False)
    df['person_day'] = df['group_id'] + "_" + df['date_human']

    # 解析 FDA1/FDB1 数值（如存在）
    if 'FDA1' in df.columns:
        df[['FDA1x','FDA1y']] = df['FDA1'].apply(safe_eval).apply(pd.Series)
    if 'FDB1' in df.columns:
        df[['FDB1x','FDB1y']] = df['FDB1'].apply(safe_eval).apply(pd.Series)

    # 按人划分
    unique_people = df['group_id'].dropna().unique()
    np.random.shuffle(unique_people)
    split_idx = int(len(unique_people) * train_ratio)
    train_people = unique_people[:split_idx]
    test_people = unique_people[split_idx:]

    train_raw = df[df['group_id'].isin(train_people)]
    test_raw  = df[df['group_id'].isin(test_people)]

    # 每人每天最多 N 条样本
    def limit_by_day(df_raw):
        return df_raw.groupby('person_day').apply(
            lambda g: g.sample(n=min(len(g), max_per_day), random_state=seed)
        ).reset_index(drop=True)

    train_df = limit_by_day(train_raw)
    test_df  = limit_by_day(test_raw)

    # 控制总样本数量（不足则有放回采样）
    def limit_sample_count(df_raw, max_samples):
        if len(df_raw) < max_samples:
            return resample(df_raw, n_samples=max_samples, replace=True, random_state=seed)
        else:
            return df_raw.sample(n=max_samples, random_state=seed)

    train_df = limit_sample_count(train_df, train_sample_limit)
    test_df  = limit_sample_count(test_df,  test_sample_limit)

    if verbose:
        print(f"最终训练集涉及人数：{train_df['group_id'].nunique()}")
        print(f"最终测试集涉及人数：{test_df['group_id'].nunique()}")

    return train_df, test_df

# ———— 主流程 ————
train_data, test_data = [], []

for file_path, label in file_list:
    if isinstance(file_path, list):
        for f in file_path:
            if not os.path.exists(f):
                raise FileNotFoundError(f"文件不存在: {f}")
        df = pd.concat([pd.read_csv(f) for f in file_path], ignore_index=True)
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        df = pd.read_csv(file_path)

    df["label"] = label

    # 与 XGB 同步的拆分方式
    train_df, test_df = split_by_person_random(
        df,
        train_sample_limit=1000,
        test_sample_limit=300,
        max_per_day=20,
        train_ratio=0.8,
        seed=42,
        verbose=True
    )

    # 打印这个类别用到的人员ID（可视化检查）
    train_df = ensure_group_id(train_df)
    test_df  = ensure_group_id(test_df)
    train_ids = sorted(train_df['group_id'].dropna().astype(str).unique())
    test_ids  = sorted(test_df['group_id'].dropna().astype(str).unique())
    print(f"[label={label}] 训练人数={len(train_ids)} -> {', '.join(train_ids)}", flush=True)
    print(f"[label={label}] 测试人数={len(test_ids)} -> {', '.join(test_ids)}",  flush=True)

    # 处理特征
    train_df, test_df = deal_file(train_df, test_df, label)

    train_data.append(train_df)
    test_data.append(test_df)

# 合并
train_df = pd.concat(train_data, ignore_index=True)
test_df  = pd.concat(test_data,  ignore_index=True)

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
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = trial.suggest_float("C", 1e-1, 1e1, log=True)
    class_weight = trial.suggest_categorical("class_weight", ["balanced"])
    gamma = None
    degree = None
    if kernel in ["rbf", "sigmoid", "poly"]:
        gamma = trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1])
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 4)

    svc_params = {"kernel": kernel, "C": C, "class_weight": class_weight, "gamma": gamma, "degree": degree}
    svc_params = {k: v for k, v in svc_params.items() if v is not None}

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svc", SVC(**svc_params))
    ])
    return cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("最优参数：", study.best_params)
print("最优交叉验证准确率：", study.best_value)

# 训练与评估
best_params = study.best_params.copy()
svc_best = SVC(
    kernel=best_params.pop("kernel"),
    C=best_params.pop("C"),
    class_weight=best_params.pop("class_weight", None),
    gamma=best_params.pop("gamma", "scale"),
    degree=best_params.pop("degree", 3)
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("svc", svc_best)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n分类报告:")
print(classification_report(y_test, y_pred))

# === Confusion Matrix: 强制保存 ===
import os, matplotlib
matplotlib.use("Agg")  # 确保是非交互后端
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

save_dir = r"C:\Temp\ml_plots"   # 用一个肯定能写的目录
os.makedirs(save_dir, exist_ok=True)
cm_png = os.path.join(save_dir, "svm_confusion_matrix.png")

print("[CM] 开始生成混淆矩阵...")
try:
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=sorted(set(y_train)),
        cmap=plt.cm.Oranges
    )
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[CM] ✅ 已保存：{cm_png}  存在? {os.path.exists(cm_png)}")
except Exception as e:
    import traceback
    print("[CM] ❌ 失败：", repr(e))
    traceback.print_exc()



# === SHAP: permutation + 小样本 + 强制保存 ===
try:
    import shap, numpy as _np
    import matplotlib.pyplot as plt

    shap_dir = r"C:\Temp\ml_plots"
    os.makedirs(shap_dir, exist_ok=True)
    shap_summary_png = os.path.join(shap_dir, "svm_shap_summary.png")
    shap_bar_png     = os.path.join(shap_dir, "svm_shap_bar.png")

    print("[SHAP] 准备生成解释 (permutation)...")

    # 采样，避免超大数据导致调试器提前中断
    rnd = _np.random.RandomState(42)
    eval_size = min(200, len(X_test))
    bg_size   = min(50,  len(X_test))
    eval_idx  = rnd.choice(len(X_test), size=eval_size, replace=False)
    bg_idx    = rnd.choice(len(X_test), size=bg_size,  replace=False)
    X_eval, X_bg = X_test.iloc[eval_idx], X_test.iloc[bg_idx]

    def _predict_proba(X):
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X)
        from sklearn.utils.extmath import softmax
        dec = pipeline.decision_function(X)
        if getattr(dec, "ndim", 1) == 1:
            p = 1 / (1 + _np.exp(-dec))
            return _np.column_stack([1 - p, p])
        else:
            return softmax(dec)

    explainer   = shap.Explainer(_predict_proba, X_bg, algorithm="permutation")
    shap_values = explainer(X_eval)
    vals = getattr(shap_values, "values", None)

    # beeswarm
    plt.figure()
    if vals is not None and vals.ndim == 3 and vals.shape[-1] >= 2:
        shap.summary_plot(shap_values[..., 1], X_eval, show=False)
    else:
        shap.summary_plot(shap_values, X_eval, show=False)
    plt.tight_layout(); plt.savefig(shap_summary_png, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[SHAP] ✅ summary：{shap_summary_png}  存在? {os.path.exists(shap_summary_png)}")

    # bar
    plt.figure()
    if vals is not None and vals.ndim == 3 and vals.shape[-1] >= 2:
        shap.summary_plot(shap_values[..., 1], X_eval, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_eval, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(shap_bar_png, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[SHAP] ✅ bar：{shap_bar_png}  存在? {os.path.exists(shap_bar_png)}")

except ImportError:
    print("[SHAP] ℹ️ 未安装 shap：conda install -c conda-forge shap")
except Exception as e:
    import traceback
    print("[SHAP] ❌ 失败：", repr(e))
    traceback.print_exc()
