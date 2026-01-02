# split.py
import numpy as np
import pandas as pd


def split_by_person_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    seed: int = 42,
    do_stratified_sample: bool = False,
    train_per_class: int = 1000,
    test_per_class: int = 300,
):
    """
    按“人”划分训练集/测试集（人不交叉）。

    新策略默认：do_stratified_sample=False
      - 不做任何抽样：训练/测试分别保留各自所有样本（满足“不要随机抽取”）
    若你将 do_stratified_sample=True：
      - 则在各自内部按 label 分层抽样（保留旧能力，便于回滚）
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    # 1) 按人划分
    persons = df[user_id_col].dropna().unique().tolist()
    rng.shuffle(persons)
    n = len(persons)
    cut = int(round(n * train_person_ratio))
    cut = max(min(cut, n - 1), 1)

    train_ids = set(persons[:cut])
    test_ids = set(persons[cut:])

    train_raw = df[df[user_id_col].isin(train_ids)].copy()
    test_raw = df[df[user_id_col].isin(test_ids)].copy()

    # 2) 可选：分层抽样
    if do_stratified_sample:
        def stratified_sample(d: pd.DataFrame, per_class: int) -> pd.DataFrame:
            parts = []
            for label, g in d.groupby(label_col):
                n_available = len(g)
                if n_available == 0:
                    continue
                n_take = min(per_class, n_available)
                parts.append(g.sample(n=n_take, random_state=seed))
            return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0].copy()

        train_final = stratified_sample(train_raw, train_per_class)
        test_final = stratified_sample(test_raw, test_per_class)
    else:
        train_final = train_raw
        test_final = test_raw

    print("=== 划分结果统计（按人） ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    if label_col in train_final.columns:
        print("训练集各类样本数:\n", train_final[label_col].value_counts(dropna=False).sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    if label_col in test_final.columns:
        print("测试集各类样本数:\n", test_final[label_col].value_counts(dropna=False).sort_index())

    return train_final, test_final
