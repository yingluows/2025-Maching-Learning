# main.py
import os
import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import joblib
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from xgboost import XGBClassifier
import shap

from deal_data import load_one_user_file, expand_bracket_features, deal_file
from split import split_by_person_stratified

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# === 超参数 ===
MAX_ROWS_PER_USER = 50  # 每个用户最多保留多少行（读取阶段限流）

# === 路径配置 ===
feature_dir = r"D:/2025_Stage/Code/XGB/ppgfeature"          # 特征文件夹
user_info_path = r"D:/2025_Stage/Code/XGB/用户列表.csv"       # 里面有“年龄”那一列
SAVE_FIG_DIR = r"D:/2025_Stage/Code/XGB/Save_fig"
SAVE_MODEL_DIR = r"D:/2025_Stage/Code/XGB/Save_model"

os.makedirs(SAVE_FIG_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)


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


def load_all_data() -> pd.DataFrame:
    """读取所有用户文件 + 打上年龄标签 + 展开 [a,b] 特征"""
    # 读用户列表
    user_info = pd.read_csv(user_info_path)
    user_info["user_id"] = user_info["user_id"].astype(str)
    user_info["user_id"] = user_info["user_id"].str.lstrip("_")  # 去掉前缀 "_"

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

        # 每个用户先限流，避免单个文件太大
        if len(df) > MAX_ROWS_PER_USER:
            df = df.sample(n=MAX_ROWS_PER_USER, random_state=42)

        all_df.append(df)

    # 合并所有用户
    df_all = pd.concat(all_df, ignore_index=True)
    df_all = expand_bracket_features(df_all)

    print("全部数据形状：", df_all.shape)
    print("用户数量：", df_all["user_id"].nunique())

    return df_all


def train_and_explain():
    # 1. 加载数据
    df_all = load_all_data()

    # 2. 按人划分 + 每类 1000/300 样本
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

    # 3. 特征清理
    train_df, test_df = deal_file(train_df, test_df)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # 4. 训练前清理无穷值为 NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # 5. Optuna 调参 —— 针对 XGBClassifier
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

    # 6. 用最优参数重新训练
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
    
    # 7. 混淆矩阵
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=sorted(y_train.unique()),
        cmap=plt.cm.Oranges
    )
    plt.title("XGB Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FIG_DIR, "xgb_confusion_matrix_optuna.png"), dpi=300)
    plt.show()

    # 8. 保存模型
    model_path = os.path.join(SAVE_MODEL_DIR, "xgb_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"模型已保存到 {model_path}")

    # 9. SHAP 解释
    sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)

    explainer = shap.Explainer(pipeline.predict, sample_X)
    shap_values = explainer(sample_X)

    plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)
    plt.title("XGB SHAP Summary (Dot)")
    plt.savefig(os.path.join(SAVE_FIG_DIR, "xgb_shap_summary_dot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    plt.title("XGB SHAP Summary (Bar)")
    plt.savefig(os.path.join(SAVE_FIG_DIR, "xgb_shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("SHAP 解释图已保存：xgb_shap_summary_dot.png 与 xgb_shap_summary_bar.png")


if __name__ == "__main__":
    train_and_explain()
