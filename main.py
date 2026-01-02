# main.py
"""
训练与评估脚本（v7策略）
- 输入：select_feature.py 输出的 train_v7.csv / test_v7.csv
- 训练：可选 XGB 或 SVM，Optuna 调参
- 训练前：可选高斯噪声上采样（仅训练集）
- 测试：输出 strict accuracy / classification report / confusion matrix
- 额外：若 CSV 包含 age，则计算 tolerant accuracy（仅评估，不参与训练）
"""

import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ========= Matplotlib 中文设置 =========
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ========= 路径配置 =========
BASE_DIR = r"D:/2025_Stage/Code/XGB"
DATA_SPLIT_DIR = os.path.join(BASE_DIR, "Data_splits")
SAVE_FIG_DIR = os.path.join(BASE_DIR, "Save_fig")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "Save_model")

os.makedirs(SAVE_FIG_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v7.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v7.csv")

FORBIDDEN_COLS = ("user_id",)  # 双保险


# ========= tolerant accuracy =========
LABEL_TO_RANGE = {
    0: (0, 20),
    1: (21, 30),
    2: (31, 40),
    3: (41, 50),
    4: (51, 60),
    5: (61, None),
}

def expanded_range(label: int, tol: int = 2):
    low, high = LABEL_TO_RANGE[int(label)]
    low2 = max(0, low - tol)
    high2 = None if high is None else (high + tol)
    return low2, high2

def tolerant_accuracy(y_pred_label, y_true_age, tol: int = 2):
    y_pred_label = np.asarray(y_pred_label).astype(int)
    y_true_age = np.asarray(y_true_age).astype(float)

    ok = np.zeros_like(y_true_age, dtype=bool)
    for i, lab in enumerate(y_pred_label):
        low2, high2 = expanded_range(lab, tol=tol)
        if high2 is None:
            ok[i] = (y_true_age[i] >= low2)
        else:
            ok[i] = (low2 <= y_true_age[i] <= high2)
    return float(ok.mean())


def load_data_from_csv(
    train_feat_path: str = TRAIN_FEAT_PATH,
    test_feat_path: str = TEST_FEAT_PATH,
    label_col: str = "label",
    age_col: str = "age",
):
    train_df = pd.read_csv(train_feat_path)
    test_df = pd.read_csv(test_feat_path)

    print("特征训练集形状：", train_df.shape)
    print("特征测试集形状：", test_df.shape)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"CSV 中必须包含列 '{label_col}' 作为类别标签。")

    # y
    y_train = train_df[label_col].astype(int)
    y_test = test_df[label_col].astype(int)

    # age（可选，仅用于 tolerant accuracy）
    y_test_age = test_df[age_col] if age_col in test_df.columns else None

    # X：drop label/age/forbidden
    drop_train = [label_col]
    if age_col in train_df.columns:
        drop_train.append(age_col)
    drop_train += [c for c in FORBIDDEN_COLS if c in train_df.columns]

    drop_test = [label_col]
    if age_col in test_df.columns:
        drop_test.append(age_col)
    drop_test += [c for c in FORBIDDEN_COLS if c in test_df.columns]

    X_train = train_df.drop(columns=drop_train, errors="ignore")
    X_test = test_df.drop(columns=drop_test, errors="ignore")

    # 双保险：防止 user_id 混入
    for c in FORBIDDEN_COLS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=[c], errors="ignore")
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c], errors="ignore")

    # inf -> nan
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    return X_train, X_test, y_train, y_test, y_test_age


def gaussian_noise_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    target: str = "max",
    sigma: float = 0.02,
    seed: int = 42,
):
    """
    高斯噪声上采样（只对数值特征有效；缺失值用均值填充用于生成样本）
    """
    rng = np.random.RandomState(seed)

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    col_means = X.mean(numeric_only=True)
    X_imp = X.copy().fillna(col_means)

    col_std = X_imp.std(numeric_only=True).replace(0, 1e-12)
    col_std_arr = col_std.values.astype(float)

    counts = y.value_counts()
    if target == "max":
        target_n = int(counts.max())
    else:
        target_n = int(target)

    X_new_list = [X_imp]
    y_new_list = [y]

    for cls, n in counts.items():
        if n >= target_n:
            continue
        need = target_n - int(n)

        idx_cls = np.where(y.values == cls)[0]
        pick_idx = rng.choice(idx_cls, size=need, replace=True)

        base = X_imp.iloc[pick_idx].to_numpy(dtype=float)
        noise = rng.normal(loc=0.0, scale=sigma * col_std_arr, size=base.shape)
        synth = base + noise

        X_synth = pd.DataFrame(synth, columns=X_imp.columns)
        y_synth = pd.Series([cls] * need)

        X_new_list.append(X_synth)
        y_new_list.append(y_synth)

    X_os = pd.concat(X_new_list, ignore_index=True)
    y_os = pd.concat(y_new_list, ignore_index=True)

    print("\n=== 高斯噪声上采样完成 ===")
    print("sigma =", sigma, "| target =", target_n)
    print("上采样前：\n", counts.sort_index())
    print("上采样后：\n", y_os.value_counts().sort_index())
    print("训练集样本数：", len(y), "->", len(y_os))

    return X_os, y_os


def create_objective(model_name: str, X_train, y_train, cv: int = 5):
    model_name = model_name.lower()

    def objective(trial):
        if model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 700),
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
            raise ValueError("model 仅支持 xgb 或 svm")

        score = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        ).mean()
        return score

    return objective


def train_and_evaluate(
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    y_test_age=None,
    n_trials: int = 30,
    tol: int = 2,
):
    model_name = model_name.lower()

    # Optuna 调参
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(model_name, X_train, y_train), n_trials=n_trials)

    print("\n=== Optuna 最优结果 ===")
    print("最优参数：", study.best_params)
    print("最优 CV accuracy：", study.best_value)

    best_params = study.best_params.copy()

    if model_name == "xgb":
        clf_best = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            **best_params,
        )
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", clf_best),
        ])
    else:
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

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 指标
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Strict Accuracy: {acc:.4f}")

    if y_test_age is not None:
        tacc = tolerant_accuracy(y_pred, y_test_age.values, tol=tol)
        print(f"✅ Tolerant Accuracy (±{tol}岁): {tacc:.4f}")

        tol_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_tolerant_accuracy.txt")
        with open(tol_path, "w", encoding="utf-8") as f:
            f.write(f"accuracy={acc:.6f}\n")
            f.write(f"tolerant_accuracy_tol_{tol}={tacc:.6f}\n")
        print(f"tolerant 指标已保存：{tol_path}")
    else:
        print("⚠ 未检测到 age 列，跳过 tolerant accuracy。")

    # 分类报告
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).T
    report_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_path, encoding="utf-8-sig", index=True)
    print(f"分类报告已保存：{report_path}")

    # 混淆矩阵
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=sorted(np.unique(y_train)),
        cmap=plt.cm.Oranges,
    )
    plt.title(f"{model_name.upper()} Confusion matrix")
    cm_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_confusion_matrix_optuna.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存：{cm_path}")

    # 保存模型
    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"模型已保存：{model_path}")

    return pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "svm"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--tol", type=int, default=2)

    parser.add_argument("--gauss_os", action="store_true", help="开启高斯噪声上采样（仅训练集）")
    parser.add_argument("--gauss_sigma", type=float, default=0.02)
    parser.add_argument("--gauss_target", type=str, default="max", help="max 或整数")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_train, X_test, y_train, y_test, y_test_age = load_data_from_csv()

    if args.gauss_os:
        tgt = args.gauss_target
        if isinstance(tgt, str) and tgt.strip().lower() != "max":
            tgt = int(tgt)
        X_train, y_train = gaussian_noise_oversample(
            X_train, y_train,
            target=tgt,
            sigma=args.gauss_sigma,
            seed=42,
        )

    train_and_evaluate(
        model_name=args.model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_test_age=y_test_age,
        n_trials=args.trials,
        tol=args.tol,
    )
