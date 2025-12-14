import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler



# ========= Matplotlib 中文设置 =========
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ========= 路径配置（按需修改） =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"

DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
SAVE_FIG_DIR = os.path.join(BASE_DIR, "Save_fig")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "Save_model")

os.makedirs(SAVE_FIG_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# 数据划分路径
TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v1.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v1.csv")


# ========= 数据读取 =========

def load_data_from_csv(
    train_feat_path: str = TRAIN_FEAT_PATH,
    test_feat_path: str = TEST_FEAT_PATH,
):
    """
    直接从 select_feature.py 导出的 train/test CSV 读取数据。
    假设这两个文件已经做完特征处理，并包含 label 列。
    """
    train_df = pd.read_csv(train_feat_path)
    test_df = pd.read_csv(test_feat_path)

    print("特征训练集形状：", train_df.shape)
    print("特征测试集形状：", test_df.shape)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # 再保险处理一下 inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    return X_train, X_test, y_train, y_test


# ========= Optuna 目标函数 =========

def create_objective(model_name: str, X_train, y_train):
    model_name = model_name.lower()

    def objective(trial):
        if model_name == "xgb":
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

            clf = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=42,
                **params,
            )

            pipeline = ImbPipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("upsample", RandomOverSampler(sampling_strategy="not majority", random_state=42)),
                ("clf", clf),
            ])


        elif model_name == "svm":
            # SVM 对尺度敏感：必须标准化
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)

            params = {"kernel": kernel, "C": C}

            if kernel in ("rbf", "poly", "sigmoid"):
                params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
            if kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)

            clf = SVC(
                probability=True,  # 便于后续 SHAP/评估输出概率
                decision_function_shape="ovr",
                random_state=42,
                **params,
            )

            pipeline = ImbPipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),                 # SVM 必须标准化
                ("upsample", RandomOverSampler(sampling_strategy="not majority",random_state=42)),
                ("clf", clf),
            ])

        else:
            raise ValueError(f"不支持的 model: {model_name}，请选择 xgb 或 svm")

        score = cross_val_score(
            pipeline, X_train, y_train,
            cv=5,
            scoring="accuracy"
        ).mean()
        return score

    return objective


# ========= 训练 / 评估 =========

def train_and_evaluate(model_name: str, X_train, X_test, y_train, y_test, n_trials: int = 30, run_shap_flag: bool = True):
    model_name = model_name.lower()

    # Optuna 调参
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(model_name, X_train, y_train), n_trials=n_trials)

    print("最优参数：", study.best_params)
    print("最优交叉验证准确率：", study.best_value)

    best_params = study.best_params.copy()

    if model_name == "xgb":
        clf_best = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            **best_params,
        )
        pipeline = ImbPipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("upsample", RandomOverSampler(sampling_strategy="not majority",random_state=42)),
            ("clf", clf_best),
        ])



    elif model_name == "svm":
        clf_best = SVC(
            probability=True,
            decision_function_shape="ovr",
            random_state=42,
            **best_params,
        )
        pipeline = ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),                 # SVM 必须标准化
            ("upsample", RandomOverSampler(sampling_strategy="not majority",random_state=42)),
            ("clf", clf_best),
        ])

    else:
        raise ValueError(f"不支持的 model: {model_name}，请选择 xgb 或 svm")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 分类报告
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).T
    report_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_path, encoding="utf-8-sig")
    print(f"分类报告已保存到: {report_path}")

    # 混淆矩阵
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=sorted(y_train.unique()),
        cmap=plt.cm.Oranges,
    )
    plt.title(f"{model_name.upper()} Confusion matrix")
    cm_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_confusion_matrix_optuna.png")
    plt.savefig(cm_path, dpi=300)
    plt.tight_layout()
    plt.show()
    print(f"混淆矩阵已保存到: {cm_path}")

    # 保存模型
    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"模型已保存到: {model_path}")

    # SHAP 模型解释
    if run_shap_flag:
        run_shap(model_name, pipeline, X_test)

    return pipeline


def run_shap(model_name: str, pipeline, X_test: pd.DataFrame):
    model_name = model_name.lower()

    # 取部分样本
    sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)

    if model_name == "xgb":
        explainer = shap.Explainer(pipeline.predict, sample_X)
        shap_values = explainer(sample_X)
        values = shap_values.values

    elif model_name == "svm":
        # KernelExplainer 相对慢：控制背景集和采样数
        background = sample_X.sample(n=min(50, len(sample_X)), random_state=42)

        # 解释概率输出（多分类时返回 n_samples x n_classes）
        explainer = shap.KernelExplainer(pipeline.predict_proba, background)
        shap_values = explainer.shap_values(sample_X, nsamples=100)

        # shap_values: list[n_classes] of (n_samples, n_features) 或 (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            # (n_classes, n_samples, n_features) -> (n_samples, n_features, n_classes)
            values = np.stack(shap_values, axis=-1)
        else:
            values = np.array(shap_values)

    else:
        raise ValueError(f"不支持的 model: {model_name}")

    # 统一：把多分类维度聚合成 (n_samples, n_features)
    if values.ndim == 3:
        values_agg = np.mean(np.abs(values), axis=2) * np.sign(np.mean(values, axis=2))
        importance = np.mean(np.abs(values), axis=(0, 2))
    else:
        values_agg = values
        importance = np.mean(np.abs(values), axis=0)

    # SHAP summary dot
    plt.figure()
    shap.summary_plot(values_agg, sample_X, show=False)
    plt.title(f"{model_name.upper()} SHAP Summary (Dot)")
    shap_dot_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_dot.png")
    plt.savefig(shap_dot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP dot 图已保存到: {shap_dot_path}")

    # SHAP bar
    plt.figure()
    shap.summary_plot(values_agg, sample_X, plot_type="bar", show=False)
    plt.title(f"{model_name.upper()} SHAP Summary (Bar)")
    shap_bar_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar 图已保存到: {shap_bar_path}")

    # 导出全部特征重要性
    importance_all = pd.DataFrame({
        "feature": sample_X.columns,
        "importance": importance,
    }).sort_values(by="importance", ascending=False)

    print("\n=== 前 20 特征 ===")
    print(importance_all.head(20))

    importance_csv_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_feature_importance.csv")
    importance_all.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
    print(f"所有特征重要性排名已保存到：{importance_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "svm"], help="选择模型：xgb 或 svm")
    parser.add_argument("--trials", type=int, default=30, help="Optuna 迭代次数")
    parser.add_argument("--no_shap", action="store_true", help="不运行 SHAP（SVM 会更快）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 直接读 select_feature 导出的 CSV
    X_train, X_test, y_train, y_test = load_data_from_csv()

    # 调参 + 训练 + 评估
    train_and_evaluate(
        model_name=args.model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_trials=args.trials,
        run_shap_flag=(not args.no_shap),
    )


