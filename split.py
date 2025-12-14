# split.py
import numpy as np
import pandas as pd


def split_by_person_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    train_per_class: int = 1000,
    test_per_class: int = 300,
    seed: int = 42,
):
    """
    按“人”划分训练集/测试集（人不交叉），并在各自内部按 label 分层抽样。
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

    # 2) 分层抽样
    def stratified_sample(d: pd.DataFrame, per_class: int) -> pd.DataFrame:
        parts = []
        for label, g in d.groupby(label_col):
            n_available = len(g)
            if n_available == 0:
                continue
            n_take = min(per_class, n_available)
            parts.append(g.sample(n=n_take, random_state=seed))
        if parts:
            return pd.concat(parts, ignore_index=True)
        else:
            return d.iloc[0:0].copy()

    train_final = stratified_sample(train_raw, train_per_class)
    test_final = stratified_sample(test_raw, test_per_class)

    print("=== 划分结果统计（按人 + 按类） ===")
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
    test_per_class: int = 2000,
    seed: int = 42,
):
    """\
    按“同一个人内部的日期窗口”划分训练集/测试集，并在各自内部按 label 分层抽样。

    规则（对每个 user）：
      - 该用户按 date_col 升序的前 train_days 个“自然日” -> 训练集
      - 剩余日期中按 date_col 升序的最后 test_days 个“自然日” -> 测试集

    边界情况：
      - 若该用户天数 <= train_days：全部进入训练集，测试集为空。
      - 若 train_days < 天数 <= train_days + test_days：前 train_days 天进训练，剩余进测试。
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    if date_col not in df.columns:
        raise ValueError(
            f"split_by_person_day_window_stratified 需要列 '{date_col}' 作为日期列，但在 df 中未找到。"
        )

    # 统一日期格式，确保可排序
    # 允许 date_col 是 'YYYY-MM-DD' 字符串或 datetime
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

    # 分层抽样（每类固定条数）
    def stratified_sample(d: pd.DataFrame, per_class: int) -> pd.DataFrame:
        parts = []
        for label, gg in d.groupby(label_col):
            n_available = len(gg)
            if n_available == 0:
                continue
            n_take = min(per_class, n_available)
            # 使用固定 seed 保证可复现；为避免每类都拿到完全相同的随机序列，混入 label 的 hash
            rs = int((seed + (hash(str(label)) % 10_000)) % (2**32 - 1))
            parts.append(gg.sample(n=n_take, random_state=rs))
        return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0].copy()

    train_final = stratified_sample(train_raw, train_per_class)
    test_final = stratified_sample(test_raw, test_per_class)

    print("=== 划分结果统计（按人前N天训练 + 后M天测试 + 按类） ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final

