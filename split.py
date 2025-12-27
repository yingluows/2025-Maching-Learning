# split_scheme1.py
import numpy as np
import pandas as pd


def _stratified_sample(
    d: pd.DataFrame,
    label_col: str,
    per_class: int | None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    在数据集 d 内部按 label 分组抽样：每个类别最多抽 per_class 条（不补齐）。
    per_class=None -> 不抽样，原样返回（用于测试集保持原分布）。
    """
    if per_class is None:
        return d.copy()

    parts = []
    for label, g in d.groupby(label_col):
        n_available = len(g)
        if n_available == 0:
            continue
        n_take = min(per_class, n_available)
        parts.append(g.sample(n=n_take, random_state=seed))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return d.iloc[0:0].copy()


def split_by_person_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    train_per_class: int = 1000,
    test_per_class: int | None = None,
    seed: int = 42,
):
    """
    方案1（与 split_data_correctly.py 对齐）：
    - 先按“人”划分 train/test（人不交叉）；
    - 只对训练集做按类抽样（train_per_class 用于控制规模/近似平衡）；
    - 测试集默认不抽样（test_per_class=None），保持原始分布。
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    persons = df[user_id_col].dropna().unique().tolist()
    rng.shuffle(persons)
    n = len(persons)
    cut = int(round(n * train_person_ratio))
    cut = max(min(cut, n - 1), 1)

    train_ids = set(persons[:cut])
    test_ids = set(persons[cut:])

    train_raw = df[df[user_id_col].isin(train_ids)].copy()
    test_raw = df[df[user_id_col].isin(test_ids)].copy()

    train_final = _stratified_sample(train_raw, label_col=label_col, per_class=train_per_class, seed=seed)
    test_final = _stratified_sample(test_raw, label_col=label_col, per_class=test_per_class, seed=seed)

    print("=== 划分结果统计（按人；训练可按类抽样；测试默认原分布） ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final


def split_by_person_day_window_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    date_col: str = "_date_for_limit",
    train_days: int = 5,
    test_days: int = 3,
    train_per_class: int = 6000,
    test_per_class: int | None = None,
    seed: int = 42,
):
    """
    方案1（对齐 split_data_correctly.py）：
    - 同人按日期窗口切分 train/test；
    - 只对训练集按类抽样；
    - 测试集默认不抽样，保持原始分布。
    """
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    if date_col not in df.columns:
        raise ValueError(
            f"split_by_person_day_window_stratified 需要列 '{date_col}' 作为日期列，但在 df 中未找到。"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    train_parts = []
    test_parts = []

    for uid, g in df.groupby(user_id_col):
        g = g.dropna(subset=[date_col]).copy()
        if g.empty:
            continue

        days = sorted(g[date_col].dt.normalize().unique().tolist())
        n_days = len(days)

        if n_days <= train_days:
            train_days_set = set(days)
            test_days_set = set()
        elif n_days <= train_days + test_days:
            train_days_set = set(days[:train_days])
            test_days_set = set(days[train_days:])
        else:
            train_days_set = set(days[:train_days])
            test_days_set = set(days[-test_days:])

        train_parts.append(g[g[date_col].dt.normalize().isin(train_days_set)])
        if test_days_set:
            test_parts.append(g[g[date_col].dt.normalize().isin(test_days_set)])

    train_raw = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    test_raw = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    train_final = _stratified_sample(train_raw, label_col=label_col, per_class=train_per_class, seed=seed)
    test_final = _stratified_sample(test_raw, label_col=label_col, per_class=test_per_class, seed=seed)

    print("=== 划分结果统计（按人前N天训练 + 后M天测试；训练可按类抽样；测试默认原分布） ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final
