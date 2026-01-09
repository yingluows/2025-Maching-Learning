# main.py
import os
import argparse
import json

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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

TRAIN_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "train_v8.csv")
TEST_FEAT_PATH = os.path.join(DATA_SPLIT_DIR, "test_v8.csv")


# ========= 年龄段定义（按你的“20岁以下、21-30、以此类推”） =========
# 默认：<=20, 21-30, 31-40, 41-50, 51-60, >=61
# 你如果有不同分段，直接改这里即可。
DEFAULT_AGE_BINS = [
    (None, 20),   # <=20
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, None),   # >=61
]


# ========= 数据读取 =========
def load_data_from_csv(train_feat_path: str = TRAIN_FEAT_PATH, test_feat_path: str = TEST_FEAT_PATH):
    train_df = pd.read_csv(train_feat_path)
    test_df = pd.read_csv(test_feat_path)

    print("特征训练集形状：", train_df.shape)
    print("特征测试集形状：", test_df.shape)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    return X_train, X_test, y_train, y_test


# ========= 高斯噪声上采样 =========
def gaussian_noise_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    target_strategy: str = "max",   # "max" or "median" or "value"
    target_value: int = None,
    noise_scale: float = 0.05,      # 噪声强度：sigma = noise_scale * feature_std
    random_state: int = 42,
):
    """
    对少数类进行高斯噪声上采样（仅训练集使用）。
    - 对每个类，用该类样本的数值特征标准差决定噪声规模
    - 生成新样本：x_new = x + N(0, sigma)
    """
    rng = np.random.RandomState(random_state)
    X = X.copy()
    y = y.copy()

    num_cols = X.columns
    Xv = X.values.astype(float)

    counts = y.value_counts()
    if target_strategy == "max":
        target_n = int(counts.max())
    elif target_strategy == "median":
        target_n = int(counts.median())
    elif target_strategy == "value":
        if target_value is None:
            raise ValueError("target_strategy='value' 时必须提供 target_value")
        target_n = int(target_value)
    else:
        raise ValueError("target_strategy 仅支持: max/median/value")

    new_X_list = [Xv]
    new_y_list = [y.values]

    for cls, n in counts.items():
        if n >= target_n:
            continue

        idx = np.where(y.values == cls)[0]
        if len(idx) == 0:
            continue

        need = target_n - n

        base = Xv[idx]
        std = np.nanstd(base, axis=0)
        std = np.where(std == 0, 1e-6, std)

        choose_idx = rng.choice(len(base), size=need, replace=True)
        samples = base[choose_idx]

        noise = rng.normal(loc=0.0, scale=(noise_scale * std), size=samples.shape)
        synth = samples + noise

        new_X_list.append(synth)
        new_y_list.append(np.full(need, cls))

    X_new = np.vstack(new_X_list)
    y_new = np.concatenate(new_y_list)

    X_new = pd.DataFrame(X_new, columns=num_cols)
    y_new = pd.Series(y_new)

    return X_new, y_new


# ========= tolerance：按“真实所在年龄段”扩展 ±tol =========
def _to_numeric(arr) -> np.ndarray:
    """把 y 转成 float 数组；不行就抛异常，避免悄悄算错。"""
    a = np.asarray(arr)
    try:
        return a.astype(float)
    except Exception as e:
        raise ValueError(
            "tolerance 需要 label / pred 可转换为数值年龄（例如 18, 25, 42）。"
            "你现在的 y 可能是类别编号或字符串。"
        ) from e


def _find_bin(age: float, bins) -> tuple[float | None, float | None]:
    """根据年龄找到所在的 bin（low, high）。low/high 为 None 表示无界。"""
    for low, high in bins:
        low_ok = True if low is None else (age >= low)
        high_ok = True if high is None else (age <= high)
        if low_ok and high_ok:
            return low, high
    # 找不到就当作“单点类”
    return age, age


def tolerant_correct_by_age_bins(y_true, y_pred, bins, tol: int = 2) -> np.ndarray:
    """
    对每个样本：
      1) 先用 y_true 找到其所属年龄段 (low, high)
      2) 把该年龄段扩展为 (low - tol, high + tol)（无界端保持无界）
      3) 若 y_pred 落在扩展区间内，则算对
    """
    yt = _to_numeric(y_true)
    yp = _to_numeric(y_pred)

    ok = np.zeros_like(yt, dtype=bool)
    for i in range(len(yt)):
        low, high = _find_bin(float(yt[i]), bins)

        # 扩展 ±tol；无界端保持无界
        low2 = None if low is None else (float(low) - tol)
        high2 = None if high is None else (float(high) + tol)

        pred = float(yp[i])
        low_ok = True if low2 is None else (pred >= low2)
        high_ok = True if high2 is None else (pred <= high2)
        ok[i] = low_ok and high_ok

    return ok


def tolerant_accuracy_by_age_bins(y_true, y_pred, bins, tol: int = 2) -> float:
    ok = tolerant_correct_by_age_bins(y_true, y_pred, bins=bins, tol=tol)
    return float(np.mean(ok)) if len(ok) else 0.0


# ========= 模型构造 =========
def build_pipeline(model_name: str, params: dict):
    model_name = model_name.lower()

    if model_name == "xgb":
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
        return pipeline

    if model_name == "svm":
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
        return pipeline

    raise ValueError("model 仅支持 xgb 或 svm")


# ========= Optuna 目标函数（带 CV + 训练折上采样） =========
def create_objective(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    oversample_noise_scale: float = 0.05,
    oversample_target: str = "max",
):
    model_name = model_name.lower()
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    def objective(trial):
        if model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
        else:
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            params = {"kernel": kernel, "C": C}
            if kernel in ("rbf", "poly", "sigmoid"):
                params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
            if kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)

        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            # 只在训练折上做高斯噪声上采样（避免泄露）
            X_tr_os, y_tr_os = gaussian_noise_oversample(
                X_tr, y_tr,
                target_strategy=oversample_target,
                noise_scale=oversample_noise_scale,
                random_state=42,
            )

            pipe = build_pipeline(model_name, params)
            pipe.fit(X_tr_os, y_tr_os)
            pred = pipe.predict(X_va)

            # Optuna 仍用“严格准确率”做优化（不污染调参）
            acc = float(np.mean(pred == y_va.values))
            scores.append(acc)

        return float(np.mean(scores))

    return objective


# ========= 训练 / 评估 =========
def train_and_evaluate(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_trials: int = 30,
    run_shap_flag: bool = True,
    oversample_noise_scale: float = 0.05,
    oversample_target: str = "max",
    use_tolerance: bool = False,
    tol: int = 2,
    age_bins=None,
):
    model_name = model_name.lower()
    age_bins = age_bins or DEFAULT_AGE_BINS

    # Optuna 调参（CV 内部训练折上采样）
    study = optuna.create_study(direction="maximize")
    study.optimize(
        create_objective(
            model_name=model_name,
            X=X_train,
            y=y_train,
            cv_splits=5,
            oversample_noise_scale=oversample_noise_scale,
            oversample_target=oversample_target,
        ),
        n_trials=n_trials,
    )

    print("最优参数：", study.best_params)
    print("最优 CV 严格准确率：", study.best_value)

    best_params = study.best_params.copy()

    # 最终训练：对整个训练集做一次上采样，再 fit
    X_train_os, y_train_os = gaussian_noise_oversample(
        X_train, y_train,
        target_strategy=oversample_target,
        noise_scale=oversample_noise_scale,
        random_state=42,
    )

    pipeline = build_pipeline(model_name, best_params)
    pipeline.fit(X_train_os, y_train_os)

    y_pred = pipeline.predict(X_test)

    # ===== 严格评估（原有报告） =====
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).T
    report_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_classification_report_strict.csv")
    df_report.to_csv(report_path, encoding="utf-8-sig")
    print(f"严格分类报告已保存到: {report_path}")

    # ===== tolerance 评估（按年龄段扩展 ±tol） =====
    if use_tolerance:
        tol_acc = tolerant_accuracy_by_age_bins(y_test.values, y_pred, bins=age_bins, tol=tol)
        tol_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_tolerance_score.txt")
        with open(tol_path, "w", encoding="utf-8") as f:
            f.write(f"tolerance_accuracy={tol_acc:.6f}\n")
            f.write(f"tol=±{tol}\n")
            f.write(f"age_bins={age_bins}\n")
        print(f"tolerance 准确率（按年龄段±{tol}岁）：{tol_acc:.6f} （已保存：{tol_path}）")

    # ===== 归一化混淆矩阵（按真实类别：行归一化）=====
    disp = ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=sorted(pd.unique(y_train)),
        normalize="true",
        cmap=plt.cm.Oranges,
        values_format=".2f",
    )

    plt.title(f"{model_name.upper()} Confusion matrix (normalized by true)")
    cm_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_confusion_matrix_optuna_norm_true.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    print(f"归一化混淆矩阵已保存到: {cm_path}")

    # 保存模型
    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"模型已保存到: {model_path}")

    # SHAP
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

    importance_csv_path = os.path.join(SAVE_FIG_DIR, f"{model_name}_shap_feature_importance.csv")
    importance_all.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
    print(f"所有特征重要性排名已保存到：{importance_csv_path}")


def parse_age_bins(s: str):
    """
    支持传 JSON，例如：
      --age_bins '[[null,20],[21,30],[31,40],[41,50],[51,60],[61,null]]'
    其中 null 表示无界。
    """
    if not s:
        return None
    bins = json.loads(s)
    out = []
    for low, high in bins:
        out.append((low, high))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "svm"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--no_shap", action="store_true")
    parser.add_argument("--noise_scale", type=float, default=0.05, help="高斯噪声强度：sigma = noise_scale * std")
    parser.add_argument("--oversample_target", type=str, default="max", choices=["max", "median"])

    # tolerance（按年龄段扩展 ±tol）
    parser.add_argument("--use_tolerance", action="store_true", help="启用 tolerance 评估（按年龄段扩展 ±tol 岁）")
    parser.add_argument("--tol", type=int, default=2, help="tolerance 年龄扩展：±tol（默认 2）")
    parser.add_argument(
        "--age_bins",
        type=str,
        default="",
        help="可选：自定义年龄段 JSON，例如 '[[null,20],[21,30],[31,40],[61,null]]'，null 表示无界"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_train, X_test, y_train, y_test = load_data_from_csv()

    bins = parse_age_bins(args.age_bins) or DEFAULT_AGE_BINS

    train_and_evaluate(
        model_name=args.model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_trials=args.trials,
        run_shap_flag=(not args.no_shap),
        oversample_noise_scale=args.noise_scale,
        oversample_target=args.oversample_target,
        use_tolerance=args.use_tolerance,
        tol=args.tol,
        age_bins=bins,
    )
