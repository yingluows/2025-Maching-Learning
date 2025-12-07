# main.py
import os
import ast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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

TRAIN_CSV_PATH = os.path.join(DATA_SPLIT_DIR, "train_raw.csv")
TEST_CSV_PATH = os.path.join(DATA_SPLIT_DIR, "test_raw.csv")


# ========= 特征列配置（含 "[a,b]" 字符串列） =========
BRACKET_FEATURE_COLS = [
    # feature_time 里的 FDA/FDB
    "FDA1", "FDB1",

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


# ========= 特征处理函数 =========

def safe_eval(val):
    """把 '[a, b]' 这类字符串安全地转成 list；错误时返回 [None, None]。"""
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
    原始 col 列暂时保留（之后在 deal_file 里统一删）
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
    - 删除各种非特征列（user_id / data_name / person_day / select_number 等）
    - 对 BRACKET_FEATURE_COLS 的 *_x, *_y 构造新特征 col_mag = sqrt(x^2 + y^2)
    - 删掉原始的字符串列 + *_x + *_y
    - 删除所有 object 列
    - 返回：仅保留数值特征 + label
    """
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

        if "label" not in df.columns:
            raise ValueError("数据中没有 label 列，请确认在调用 deal_file 前已经打好标签。")

        # 1. 删各种非特征列
        df = df.drop(
            columns=[c for c in drop_not_features if c in df.columns],
            errors="ignore",
        )

        # 2. 用 *_x, *_y 构造合成特征 col_mag
        for col in BRACKET_FEATURE_COLS:
            cx = f"{col}_x"
            cy = f"{col}_y"
            if cx in df.columns and cy in df.columns:
                df[f"{col}_mag"] = np.sqrt(df[cx] ** 2 + df[cy] ** 2)

        # 3. 删掉原始 "[a,b]" 字符串列
        df = df.drop(
            columns=[c for c in BRACKET_FEATURE_COLS if c in df.columns],
            errors="ignore",
        )

        # 4. 分离 label
        label = df["label"]
        feature_df = df.drop(columns=["label"])

        # 5. 删掉所有 *_x, *_y，只保留 *_mag 和其他数值列
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")],
            errors="ignore",
        )

        # 6. 删除所有 object 列（只保留数值特征）
        feature_df = feature_df.select_dtypes(include=[np.number])

        print(f"[deal_file] 处理后保留特征数：{feature_df.shape[1]}")

        df_processed = pd.concat([feature_df, label], axis=1)
        return df_processed

    train_df = _process(train_df)
    test_df = _process(test_df)

    return train_df, test_df


# ========= 数据加载 + 处理 =========

def load_and_process_data(
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH,
):
    # 读取 raw 划分结果（含所有原始列）
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("原始训练集形状：", train_df.shape)
    print("原始测试集形状：", test_df.shape)

    # 展开 "[a,b]" 字符串列
    train_df = expand_bracket_features(train_df)
    test_df = expand_bracket_features(test_df)

    # 统一做特征清理
    train_df, test_df = deal_file(train_df, test_df)

    # 分离 X/y
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # 替换 inf 为 NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    print("处理后训练集形状：", X_train.shape)
    print("处理后测试集形状：", X_test.shape)

    return X_train, X_test, y_train, y_test


# ========= 训练 / 调参 / 评估 =========

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
            **params,
        )

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("xgb", xgb),
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

    best_params = study.best_params.copy()
    xgb_best = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        **best_params,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("xgb", xgb_best),
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
        cmap=plt.cm.Oranges,
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

    # SHAP 模型解释
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
    plt.savefig(shap_dot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP dot 图已保存到: {shap_dot_path}")

    # SHAP bar
    plt.figure()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    plt.title("XGB SHAP Summary (Bar)")
    shap_bar_path = os.path.join(SAVE_FIG_DIR, "xgb_shap_summary_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar 图已保存到: {shap_bar_path}")

    # 导出全部特征重要性
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    importance_all = pd.DataFrame({
        "feature": sample_X.columns,
        "importance": shap_importance,
    }).sort_values(by="importance", ascending=False)

    print("\n=== 特征排名（显示前 20 个） ===")
    print(importance_all.head(20))

    importance_csv_path = os.path.join(SAVE_FIG_DIR, "xgb_shap_feature_importance.csv")
    importance_all.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
    print(f"所有特征重要性排名已保存到：{importance_csv_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_process_data()
    train_and_evaluate(X_train, X_test, y_train, y_test)
