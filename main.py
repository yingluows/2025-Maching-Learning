# main.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import optuna
import shap

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# 中文字体设置（按需修改）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 路径配置 =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"

DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
SAVE_FIG_DIR = os.path.join(BASE_DIR, "Save_fig")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "Save_model")

os.makedirs(SAVE_FIG_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(DATA_SPLIT_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_SPLIT_DIR, "test.csv")


def load_data(train_path: str = TRAIN_CSV_PATH,
              test_path: str = TEST_CSV_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # 替换 inf 为 NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    print("训练集形状：", X_train.shape)
    print("测试集形状：", X_test.shape)

    return X_train, X_test, y_train, y_test


def create_objective(X_train, y_train):
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

        score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
        return score

    return objective


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Optuna 调参
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(X_train, y_train), n_trials=30)

    print("最优参数：", study.best_params)
    print("最优交叉验证准确率：", study.best_value)

    # 使用最优参数训练模型
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

    # 分类报告
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).T
    report_path = os.path.join(SAVE_FIG_DIR, "classification_report.csv")
    df_report.to_csv(report_path, encoding="utf-8-sig")
    print(f"分类报告已保存到: {report_path}")

    # 混淆矩阵
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=sorted(y_train.unique()),
        cmap=plt.cm.Oranges
    )
    plt.title("XGB Confusion matrix")
    cm_path = os.path.join(SAVE_FIG_DIR, "xgb_confusion_matrix_optuna.png")
    plt.savefig(cm_path, dpi=300)
    plt.tight_layout()
    plt.show()
    print(f"混淆矩阵已保存到: {cm_path}")

    # 保存模型
    model_path = os.path.join(SAVE_MODEL_DIR, "xgb_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"模型已保存到: {model_path}")

    # SHAP 模型解释（可选）
    run_shap(pipeline, X_test)

    return pipeline


def run_shap(pipeline, X_test: pd.DataFrame):
    # 取部分样本
    sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)

    explainer = shap.Explainer(pipeline.predict, sample_X)
    shap_values = explainer(sample_X)

    # SHAP summary dot
    plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)
    plt.title("XGB SHAP Summary (Dot)")
    shap_dot_path = os.path.join(SAVE_FIG_DIR, "xgb_shap_summary_dot.png")
    plt.savefig(shap_dot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP dot 图已保存到: {shap_dot_path}")

    # SHAP bar
    plt.figure()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    plt.title("XGB SHAP Summary (Bar)")
    shap_bar_path = os.path.join(SAVE_FIG_DIR, "xgb_shap_summary_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar 图已保存到: {shap_bar_path}")

    # 导出全部特征重要性
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    importance_all = pd.DataFrame({
        "feature": sample_X.columns,
        "importance": shap_importance
    }).sort_values(by="importance", ascending=False)

    print("\n=== 特征排名（显示前 20 个） ===")
    print(importance_all.head(20))

    importance_csv_path = os.path.join(SAVE_FIG_DIR, "xgb_shap_feature_importance.csv")
    importance_all.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
    print(f"所有特征重要性排名已保存到：{importance_csv_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    train_and_evaluate(X_train, X_test, y_train, y_test)

