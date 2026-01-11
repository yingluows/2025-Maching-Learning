# -*- coding: utf-8 -*-
import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


# ========= Matplotlib 中文设置 =========
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ========= 默认保存路径（相对 main.py 所在目录） =========
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FIG_DIR = os.path.join(_THIS_DIR, "Save_fig")
SAVE_MODEL_DIR = os.path.join(_THIS_DIR, "Save_model")
os.makedirs(SAVE_FIG_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)


# ========= 年龄分箱：20-29, 30-39, ..., 70-79 =========
# label -> (low, high)，均为闭区间
LABEL_TO_RANGE = {
    0: (20, 29),
    1: (30, 39),
    2: (40, 49),
    3: (50, 59),
    4: (60, 69),
    5: (70, 79),
}
CLASS_NAMES = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]


def age_to_label(age: float) -> int:
    """把真实年龄映射到 6 个类别（仅允许 20-79）。"""
    age = int(age)
    if age < 20 or age > 79:
        raise ValueError("age_to_label 只接受 20-79 岁（含）范围内的年龄。")
    return (age - 20) // 10  # 20-29->0 ... 70-79->5


# ========= 放宽判对：tolerant accuracy（旧策略） =========
def expanded_range(label: int, tol: int = 2):
    """给定预测label，返回放宽后的可接受年龄区间 [low, high]（闭区间）。"""
    low, high = LABEL_TO_RANGE[int(label)]
    return low - tol, high + tol


def tolerant_accuracy(y_pred_label, y_true_age, tol: int = 2) -> float:
    """
    判定：真实年龄落在“预测label的放宽区间”内 => 算对
    y_pred_label: 预测label（0-5）
    y_true_age:   真实年龄
    """
    y_pred_label = np.asarray(y_pred_label).astype(int)
    y_true_age = np.asarray(y_true_age).astype(float)

    ok = np.zeros_like(y_true_age, dtype=bool)
    for i, lab in enumerate(y_pred_label):
        low2, high2 = expanded_range(lab, tol=tol)
        ok[i] = (low2 <= y_true_age[i] <= high2)

    return float(ok.mean())


# ========= 数据读取（训练集/测试集 CSV） =========
def load_data_from_csv(
    train_path: str,
    test_path: str,
    age_col: str = "年龄",
    drop_cols_extra=None,
):
    """
    读取训练/测试 CSV，并做以下处理：
    1) 去掉 20-79 岁以外样本（训练集和测试集都做）
    2) 根据年龄生成 6 类 label：20-29, 30-39, 40-49, 50-59, 60-69, 70-79
    3) X 中默认去掉 age_col，以及常见的 user_id（如存在），避免泄露
    """
    drop_cols_extra = drop_cols_extra or []

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if age_col not in train_df.columns or age_col not in test_df.columns:
        raise ValueError(f"CSV 中必须包含真实年龄列 '{age_col}'。")

    # 确保年龄是数值
    train_df = train_df[pd.to_numeric(train_df[age_col], errors="coerce").notna()].copy()
    test_df = test_df[pd.to_numeric(test_df[age_col], errors="coerce").notna()].copy()
    train_df[age_col] = train_df[age_col].astype(int)
    test_df[age_col] = test_df[age_col].astype(int)

    # 只保留 20-79 岁
    train_df = train_df[(train_df[age_col] >= 20) & (train_df[age_col] <= 79)].copy()
    test_df = test_df[(test_df[age_col] >= 20) & (test_df[age_col] <= 79)].copy()

    print("训练集(过滤20-79后)形状：", train_df.shape)
    print("测试集(过滤20-79后)形状：", test_df.shape)

    # 生成 label
    train_df["label"] = train_df[age_col].apply(age_to_label).astype(int)
    test_df["label"] = test_df[age_col].apply(age_to_label).astype(int)

    # y（类别）与 y_age（真实年龄）
    y_train = train_df["label"]
    y_test = test_df["label"]
    y_train_age = train_df[age_col].astype(float)
    y_test_age = test_df[age_col].astype(float)

    # X：去掉 label + age + user_id（若存在）+ 额外指定列
    drop_cols = ["label", age_col] + list(drop_cols_extra)
    if "user_id" in train_df.columns:
        drop_cols.append("user_id")

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

    # inf -> nan
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # 打印类别分布（显示为类别名）
    idx_map = {i: CLASS_NAMES[i] for i in range(6)}
    print("\n=== 类别分布（训练集）===")
    print(y_train.value_counts().sort_index().rename(index=idx_map))
    print("\n=== 类别分布（测试集）===")
    print(y_test.value_counts().sort_index().rename(index=idx_map))

    return X_train, X_test, y_train, y_test, y_train_age, y_test_age


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
                num_class=6,
                n_jobs=-1,
                random_state=42,
                tree_method="hist",
                **params,
            )

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("clf", clf),
            ])

        elif model_name == "svm":
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)

            params = {"kernel": kernel, "C": C}

            if kernel in ("rbf", "poly", "sigmoid"):
                params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
            if kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)

            clf = SVC(
                probability=True,
                decision_function_shape="ovr",
                random_state=42,
                **params,
            )

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
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


def save_permutation_importance(
    pipe,
    X,
    y,
    feature_names,
    out_csv,
    n_repeats=10,
    random_state=42,
    scoring="f1_macro"
):
    """
    适用于 SVM / 任意模型的特征重要性（模型无关、推荐）
    """
    result = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Permutation importance 已保存到: {out_csv}")


def run_shap(model_name: str, pipeline, X_test: pd.DataFrame):
    model_name = model_name.lower()

    # 抽样，避免太慢
    sample_X_df = X_test.sample(n=min(200, len(X_test)), random_state=42)
    feature_names = sample_X_df.columns.tolist()

    if model_name == "xgb":
        # 这里用 shap.Explainer + predict_proba，兼容性最好
        explainer = shap.Explainer(pipeline.predict_proba, sample_X_df)
        shap_values = explainer(sample_X_df)
        values = shap_values.values  # (n, p, K) 或 (n, p)

        # 多分类聚合：按类取绝对值平均 -> (n,p)
        if values.ndim == 3:
            values_agg = np.mean(values, axis=2)
            importance = np.mean(np.abs(values), axis=(0, 2))
        else:
            values_agg = values
            importance = np.mean(np.abs(values), axis=0)

        # dot
        plt.figure()
        shap.summary_plot(values_agg, sample_X_df, show=False)
        plt.title(f"{model_name.upper()} SHAP Summary (Dot)")
        shap_dot_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_dot.png")
        plt.savefig(shap_dot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP dot 图已保存到: {shap_dot_path}")

        # bar
        plt.figure()
        shap.summary_plot(values_agg, sample_X_df, plot_type="bar", show=False)
        plt.title(f"{model_name.upper()} SHAP Summary (Bar)")
        shap_bar_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_bar.png")
        plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP bar 图已保存到: {shap_bar_path}")

        importance_all = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values(by="importance", ascending=False)

        print("\n=== 前 20 特征（SHAP） ===")
        print(importance_all.head(20))

        importance_csv_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_feature_importance.csv")
        importance_all.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
        print(f"所有 SHAP 特征重要性已保存到：{importance_csv_path}")
        return

    elif model_name == "svm":
        # ✅ 关键修复：KernelExplainer 用 numpy + lambda，避免触发 Pipeline.feature_names_in_ 的 setter 问题
        sample_X = sample_X_df.to_numpy()
        background_df = sample_X_df.sample(n=min(50, len(sample_X_df)), random_state=42)
        background = background_df.to_numpy()

        f = lambda x: pipeline.predict_proba(x)  # 不要直接传 pipeline/predict_proba 引用给 SHAP 的某些包装路径
        explainer = shap.KernelExplainer(f, background)

        shap_values = explainer.shap_values(sample_X, nsamples=100)

        # 多分类：list[K] each (n, p) -> (n,p,K)
        if isinstance(shap_values, list):
            values = np.stack(shap_values, axis=-1)
        else:
            values = np.array(shap_values)

        # 聚合
        if values.ndim == 3:
            values_agg = np.mean(values, axis=2)         # (n,p)
            importance = np.mean(np.abs(values), axis=(0, 2))
        else:
            values_agg = values
            importance = np.mean(np.abs(values), axis=0)

        # dot（这里 features 用 numpy，并显式传 feature_names）
        plt.figure()
        shap.summary_plot(values_agg, sample_X, feature_names=feature_names, show=False)
        plt.title(f"{model_name.upper()} SHAP Summary (Dot)")
        shap_dot_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_dot.png")
        plt.savefig(shap_dot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP dot 图已保存到: {shap_dot_path}")

        # bar
        plt.figure()
        shap.summary_plot(values_agg, sample_X, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"{model_name.upper()} SHAP Summary (Bar)")
        shap_bar_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_bar.png")
        plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP bar 图已保存到: {shap_bar_path}")

        importance_all = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values(by="importance", ascending=False)

        print("\n=== 前 20 特征（SHAP） ===")
        print(importance_all.head(20))

        importance_csv_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_feature_importance.csv")
        importance_all.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
        print(f"所有 SHAP 特征重要性已保存到：{importance_csv_path}")
        return

    else:
        raise ValueError(f"不支持的 model: {model_name}")



# ========= 训练 / 评估 =========
def train_and_evaluate(
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    y_test_age=None,
    n_trials: int = 30,
    run_shap_flag: bool = True,
    tol: int = 2,
):
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
            num_class=6,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
            **best_params,
        )
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", clf_best),
        ])

    elif model_name == "svm":
        clf_best = SVC(
            probability=True,
            decision_function_shape="ovr",
            random_state=42,
            **best_params,
        )
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", clf_best),
        ])
    else:
        raise ValueError(f"不支持的 model: {model_name}，请选择 xgb 或 svm")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # ====== SVM 特征重要性（Permutation Importance）======
    if model_name == "svm":
        pi_csv = os.path.join(SAVE_FIG_DIR, "svm_permutation_importance.csv")
        save_permutation_importance(
            pipe=pipeline,
            X=X_test,
            y=y_test,
            feature_names=X_test.columns.tolist(),
            out_csv=pi_csv,
            scoring="f1_macro"
        )

    # ====== 普通 accuracy ======
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ 普通 Accuracy: {acc:.4f}")

    # ====== 放宽判对 tolerant accuracy ======
    if y_test_age is not None:
        tacc = tolerant_accuracy(y_pred_label=y_pred, y_true_age=y_test_age.values, tol=tol)
        print(f"✅ Tolerant Accuracy (±{tol}岁): {tacc:.4f}")

        tol_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_tolerant_accuracy.txt")
        with open(tol_path, "w", encoding="utf-8") as f:
            f.write(f"accuracy={acc:.6f}\n")
            f.write(f"tolerant_accuracy_tol_{tol}={tacc:.6f}\n")
        print(f"放宽判对指标已保存到: {tol_path}")

    # 分类报告（带类别名）
    report_dict = classification_report(
        y_test, y_pred,
        labels=list(range(6)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report_dict).T
    report_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_path, encoding="utf-8-sig", index=True)
    print(f"分类报告已保存到: {report_path}")

    # 混淆矩阵
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=CLASS_NAMES,
        cmap=plt.cm.Blues,
    )
    plt.title(f"{model_name.upper()} Confusion matrix")
    cm_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_confusion_matrix_optuna.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存到: {cm_path}")

    # 保存模型
    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"模型已保存到: {model_path}")

    # SHAP 模型解释
    if run_shap_flag:
        run_shap(model_name, pipeline, X_test)

    return pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "svm"], help="选择模型：xgb 或 svm")
    parser.add_argument("--trials", type=int, default=30, help="Optuna 迭代次数")
    parser.add_argument("--no_shap", action="store_true", help="不运行 SHAP（SVM 会更快）")
    parser.add_argument("--tol", type=int, default=2, help="放宽判对的年龄容忍度（±tol岁），默认 2")

    parser.add_argument("--train_csv", type=str, default=os.path.join(_THIS_DIR, "训练集_修正版.csv"),
                        help="训练集 CSV 路径（默认同目录下：训练集_修正版.csv）")
    parser.add_argument("--test_csv", type=str, default=os.path.join(_THIS_DIR, "测试集_修正版.csv"),
                        help="测试集 CSV 路径（默认同目录下：测试集_修正版.csv）")
    parser.add_argument("--age_col", type=str, default="年龄", help="CSV 中真实年龄列名，默认：年龄")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_train, X_test, y_train, y_test, y_train_age, y_test_age = load_data_from_csv(
        train_path=args.train_csv,
        test_path=args.test_csv,
        age_col=args.age_col,
    )

    train_and_evaluate(
        model_name=args.model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_test_age=y_test_age,
        n_trials=args.trials,
        run_shap_flag=(not args.no_shap),
        tol=args.tol,
    )
