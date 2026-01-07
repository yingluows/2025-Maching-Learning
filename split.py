# split.py
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
    per_class=None -> 不抽样，原样返回。
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
    train_per_class: int | None = None,
    test_per_class: int | None = None,
    seed: int = 42,
):
    """
    - 先按“人”划分 train/test（人不交叉）
    - 可选：对训练集/测试集按类做“最多抽 N 条”（不补齐，主要用于控规模）
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    persons = df[user_id_col].dropna().unique().tolist()
    rng.shuffle(persons)
    n = len(persons)
    if n < 2:
        raise ValueError("可用用户数 < 2，无法按人划分 train/test。")

    cut = int(round(n * train_person_ratio))
    cut = max(min(cut, n - 1), 1)

    train_ids = set(persons[:cut])
    test_ids = set(persons[cut:])

    train_raw = df[df[user_id_col].isin(train_ids)].copy()
    test_raw = df[df[user_id_col].isin(test_ids)].copy()

    train_final = _stratified_sample(train_raw, label_col=label_col, per_class=train_per_class, seed=seed)
    test_final = _stratified_sample(test_raw, label_col=label_col, per_class=test_per_class, seed=seed)

    print("=== 按人划分结果 ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final
