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
from xgboost import XGBClassifier


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

# 数据划分路径（select_feature.py 导出的特征）
TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v7.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v7.csv")


# ========= 放宽判对：tolerant accuracy =========
# label -> (low, high)，high=None 表示无穷大
LABEL_TO_RANGE = {
    0: (0, 20),
    1: (21, 30),
    2: (31, 40),
    3: (41, 50),
    4: (51, 60),
    5: (61, None),
}

def expanded_range(label: int, tol: int = 2):
    """给定预测label，返回放宽后的可接受年龄区间 [low, high]；high=None 表示无穷大。"""
    low, high = LABEL_TO_RANGE[int(label)]
    low2 = max(0, low - tol)
    high2 = None if high is None else (high + tol)
    return low2, high2

def tolerant_accuracy(y_pred_label, y_true_age, tol: int = 2):
    """
    判定：真实年龄落在“预测label的放宽区间”内 => 算对
    y_pred_label: 预测label
    y_true_age:   真实年龄（必须是年龄，不是label）
    """
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


# ========= 数据读取 =========
def load_data_from_csv(
    train_feat_path: str = TRAIN_FEAT_PATH,
    test_feat_path: str = TEST_FEAT_PATH,
    label_col: str = "label",
    age_col: str = "age",
):
    """
    从 select_feature.py 导出的 train/test CSV 读取数据。
    - 必须包含 label
    - 若包含 age，则用于 tolerant accuracy（放宽判对）
    """
    train_df = pd.read_csv(train_feat_path)
    test_df = pd.read_csv(test_feat_path)

    print("特征训练集形状：", train_df.shape)
    print("特征测试集形状：", test_df.shape)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"CSV 中必须包含列 '{label_col}' 作为类别标签。")

    # 取出 y
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    # 取出 age（如果有）
    y_train_age = train_df[age_col] if age_col in train_df.columns else None
    y_test_age = test_df[age_col] if age_col in test_df.columns else None

    # X：去掉 label 和 age（age 不作为特征喂给模型，避免泄露）
    drop_cols = [label_col]
    if age_col in train_df.columns:
        drop_cols.append(age_col)

    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    # 再保险处理一下 inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    if y_test_age is None:
        print(f"⚠ 提示：test CSV 中没有 '{age_col}' 列，将跳过 tolerant accuracy 计算。")
    else:
        print(f"✅ 检测到 '{age_col}' 列：将额外计算 tolerant accuracy（放宽判对）。")

    return X_train, X_test, y_train, y_test, y_train_age, y_test_age

def gaussian_noise_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    target: str = "max",   # "max" 或 int
    sigma: float = 0.02,   # 噪声强度：按每列std缩放
    seed: int = 42,
):
    """
    训练集高斯噪声上采样：
    - 对每个少数类，补到 target 数量（target="max" 表示补到多数类数量）
    - 通过“抽取原样本 + 加 N(0, sigma*std)”生成新样本
    - 只对训练集用，测试集绝对不要用
    """
    rng = np.random.RandomState(seed)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 为了生成噪声，先把 NaN 用列均值填掉（只用于造样本；你后面 pipeline 里 imputer 仍可保留）
    col_means = X.mean(numeric_only=True)
    X_imp = X.copy()
    X_imp = X_imp.fillna(col_means)

    # 每列 std（用于缩放噪声），std=0 的列给一个很小的值避免全0
    col_std = X_imp.std(numeric_only=True).replace(0, 1e-12)
    col_std_arr = col_std.values.astype(float)

    # 目标数量
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

    # 打印上采样前后分布
    print("\n=== 高斯噪声上采样完成 ===")
    print("sigma =", sigma, "| target =", target_n)
    print("上采样前：\n", counts.sort_index())
    print("上采样后：\n", y_os.value_counts().sort_index())
    print("训练集样本数：", len(y), "->", len(y_os))

    return X_os, y_os


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
            n_jobs=-1,
            random_state=42,
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

    # ====== 普通 accuracy ======
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ 普通 Accuracy: {acc:.4f}")

    # ====== 放宽判对 tolerant accuracy ======
    if y_test_age is not None:
        tacc = tolerant_accuracy(y_pred_label=y_pred, y_true_age=y_test_age.values, tol=tol)
        print(f"✅ Tolerant Accuracy (±{tol}岁): {tacc:.4f}")

        # 保存到文件
        tol_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_tolerant_accuracy.txt")
        with open(tol_path, "w", encoding="utf-8") as f:
            f.write(f"accuracy={acc:.6f}\n")
            f.write(f"tolerant_accuracy_tol_{tol}={tacc:.6f}\n")
        print(f"放宽判对指标已保存到: {tol_path}")

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

    sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)

    if model_name == "xgb":
        explainer = shap.Explainer(pipeline.predict, sample_X)
        shap_values = explainer(sample_X)
        values = shap_values.values

    elif model_name == "svm":
        background = sample_X.sample(n=min(50, len(sample_X)), random_state=42)
        explainer = shap.KernelExplainer(pipeline.predict_proba, background)
        shap_values = explainer.shap_values(sample_X, nsamples=100)

        if isinstance(shap_values, list):
            values = np.stack(shap_values, axis=-1)
        else:
            values = np.array(shap_values)

    else:
        raise ValueError(f"不支持的 model: {model_name}")

    if values.ndim == 3:
        values_agg = np.mean(np.abs(values), axis=2) * np.sign(np.mean(values, axis=2))
        importance = np.mean(np.abs(values), axis=(0, 2))
    else:
        values_agg = values
        importance = np.mean(np.abs(values), axis=0)

    plt.figure()
    shap.summary_plot(values_agg, sample_X, show=False)
    plt.title(f"{model_name.upper()} SHAP Summary (Dot)")
    shap_dot_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_dot.png")
    plt.savefig(shap_dot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP dot 图已保存到: {shap_dot_path}")

    plt.figure()
    shap.summary_plot(values_agg, sample_X, plot_type="bar", show=False)
    plt.title(f"{model_name.upper()} SHAP Summary (Bar)")
    shap_bar_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_summary_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar 图已保存到: {shap_bar_path}")

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
    parser.add_argument("--tol", type=int, default=2, help="放宽判对的年龄容忍度（±tol岁），默认 2")
    parser.add_argument("--age_col", type=str, default="age", help="CSV 中真实年龄列名，默认 age")
    parser.add_argument("--gauss_os", action="store_true", help="开启高斯噪声上采样（仅训练集）")
    parser.add_argument("--gauss_sigma", type=float, default=0.02, help="高斯噪声强度 sigma（默认 0.02）")
    parser.add_argument("--gauss_target", type=str, default="max", help="上采样目标：max 或整数(如 5000)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 读 select_feature 导出的 CSV
    X_train, X_test, y_train, y_test, y_train_age, y_test_age = load_data_from_csv(
        age_col=args.age_col
    )

    # ====== 高斯噪声上采样（仅训练集） ======
    if args.gauss_os:
        tgt = args.gauss_target
        if isinstance(tgt, str) and tgt.strip().lower() != "max":
            try:
                tgt = int(tgt)
            except Exception:
                raise ValueError("--gauss_target 只能是 'max' 或整数，例如 5000")

        X_train, y_train = gaussian_noise_oversample(
            X_train, y_train,
            target=tgt,
            sigma=args.gauss_sigma,
            seed=42
        )


    # 调参 + 训练 + 评估
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
