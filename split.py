# split.py —— 兼容“user_id/证件号码”与旧 feature_name 的七种划分策略

import numpy as np
import pandas as pd
from sklearn.utils import resample

# ========== 基础工具：尽量不改动原调用方 ==========
def _prepare_group_and_time(df: pd.DataFrame,
                            id_candidates=("user_id", "证件号码", "group_id"),
                            date_candidates=("date", "日期", "record_date", "测量日期"),
                            time_candidates=("datetime", "timestamp", "测量时间")) -> pd.DataFrame:
    """
    统一生成/对齐以下列：
      - group_id   ：优先 user_id/证件号码/group_id；否则从 feature_name 解析；否则缺失
      - date_human ：优先现成日期列（到日）；否则从 data_name/feature_name 解析；否则缺失
      - date_obj   ：date_human 对应的 datetime（可空）
      - group_time ：优先现成时间戳列；否则从 data_name/feature_name 解析；否则用行号占位
      - person_day ：group_id + "_" + date_human（可空）
    """
    df = df.copy()

    # ------ group_id ------
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col:
        df["group_id"] = df[id_col].astype(str)
    elif "feature_name" in df.columns:
        df["group_id"] = df["feature_name"].astype(str).str.extract(r"^(\d+)", expand=False)
    else:
        df["group_id"] = pd.NA

    # ------ date_human / date_obj ------
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is not None:
        d = pd.to_datetime(df[date_col], errors="coerce")
        df["date_human"] = d.dt.strftime("%Y_%m_%d")
        df["date_obj"] = d.dt.normalize()

    elif "data_name" in df.columns:
        # 解析形如 xxx_YYYYMMDDhhmmss
        ts_str = df["data_name"].astype(str).str.extract(r"_(\d{14})$", expand=False)
        d = pd.to_datetime(ts_str, format="%Y%m%d%H%M%S", errors="coerce")
        df["date_human"] = d.dt.strftime("%Y_%m_%d")
        df["date_obj"] = d.dt.normalize()

    elif "feature_name" in df.columns:
        df["date_human"] = df["feature_name"].astype(str).str.extract(
            r"^\d+_(\d+_\d+_\d+)", expand=False
        )
        compact = df["feature_name"].astype(str).str.extract(
            r"_(\d{8})\d{6}_\d+$", expand=False
        )
        df["date_obj"] = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")

    else:
        df["date_human"] = pd.NA
        df["date_obj"] = pd.NaT

    # ------ group_time ------
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is not None:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["group_time"] = t.dt.strftime("%Y%m%d%H%M%S").fillna(df.index.astype(str))

    elif "data_name" in df.columns:
        ts_str = df["data_name"].astype(str).str.extract(r"_(\d{14})$", expand=False)
        df["group_time"] = ts_str.fillna(df.index.astype(str))

    elif "feature_name" in df.columns:
        df["group_time"] = df["feature_name"].astype(str).str.extract(
            r"_(\d{14})_\d+$", expand=False
        )
        df["group_time"] = df["group_time"].fillna(df.index.astype(str))

    else:
        df["group_time"] = df.index.astype(str)

    # ------ person_day ------
    df["person_day"] = df["group_id"].astype(str) + "_" + df["date_human"].astype(str)
    return df




# ========== 1) 固定行号范围 ==========
def split_by_fixed_index(df: pd.DataFrame,
                         train_range: tuple = (0, 2100),
                         test_range: tuple = (2200, 3000),
                         min_required_rows: int = 3000,
                         verbose: bool = True):
    """
    与原版一致：按行号切片；仅补充 group_id 方便后续统计。
    """
    if len(df) < min_required_rows:
        if verbose:
            print(f"⚠️ 文件数据不足 {min_required_rows} 条，跳过")
        return None, None

    df = _prepare_group_and_time(df)
    train_df = df.iloc[train_range[0]:train_range[1]].copy()
    test_df  = df.iloc[test_range[0]:test_range[1]].copy()
    return train_df, test_df


# ========== 2) 随机按人拆分，并限制每人每天(或每人)样本数 + 总样本数 ==========
def split_by_person_random(
    df,
    user_id_col: str,
    label_col: str,
    train_ratio: float = 0.8,
    max_per_day: int | None = None,
    seed: int = 42,
    train_sample_limit: int | None = None,
    test_sample_limit: int | None = None,
):
    import numpy as np
    import pandas as pd

    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)
    rng = np.random.default_rng(seed)

    # === 1) 每人每天样本数限制 ===
    if max_per_day and max_per_day > 0:
        if "data_name" in df.columns:
            # 从 data_name 提取日期（到日）
            ts_str = df["data_name"].astype(str).str.extract(r"_(\d{8})\d{6}$", expand=False)
            date_from_name = pd.to_datetime(ts_str, format="%Y%m%d", errors="coerce")
            df["_date_from_name"] = date_from_name.dt.strftime("%Y-%m-%d")

            df = (
                df.sort_values([user_id_col, "_date_from_name"])
                  .groupby([user_id_col, "_date_from_name"], as_index=False)
                  .head(max_per_day)
            )

        elif "date" in df.columns:
            df = (df.sort_values([user_id_col, "date"])
                    .groupby([user_id_col, "date"], as_index=False)
                    .head(max_per_day))
        elif "datetime" in df.columns:
            df = (df.sort_values([user_id_col, "datetime"])
                    .groupby([user_id_col, "datetime"], as_index=False)
                    .head(max_per_day))
        else:
            # 实在没有任何时间信息，就按“每人最多 N 条”处理
            df = (df.sort_values([user_id_col])
                    .groupby(user_id_col, as_index=False)
                    .head(max_per_day))

    # === 2) 按人随机拆分（后面保持不变） ===
    persons = df[user_id_col].dropna().astype(str).unique().tolist()
    rng.shuffle(persons)
    n = len(persons)
    cut = int(round(n * float(train_ratio)))
    cut = max(min(cut, n), 0)

    train_ids = set(persons[:cut])
    test_ids  = set(persons[cut:])

    train_df = df[df[user_id_col].isin(train_ids)].copy()
    test_df  = df[df[user_id_col].isin(test_ids)].copy()

    if train_df.empty or test_df.empty:
        cut = max(min(int(round(n * 0.8)), n), 1)
        train_ids = set(persons[:cut])
        test_ids  = set(persons[cut:])
        train_df = df[df[user_id_col].isin(train_ids)].copy()
        test_df  = df[df[user_id_col].isin(test_ids)].copy()

    # 3) 总样本数上限
    if train_sample_limit and len(train_df) > train_sample_limit:
        train_df = train_df.sample(n=train_sample_limit, random_state=seed)
    if test_sample_limit and len(test_df) > test_sample_limit:
        test_df = test_df.sample(n=test_sample_limit, random_state=seed)

    return train_df, test_df


def split_by_person_stratified(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    label_col: str = "label",
    train_person_ratio: float = 0.8,
    max_per_person: int = 30,
    train_per_class: int = 1000,
    test_per_class: int = 300,
    seed: int = 42,
):
    """
    按人划分 + 每人最多 max_per_person 条 + 每类固定样本数（1000 / 300）
    1. 按 user_id 随机划分 train/test 人群
    2. 各自内部，每人最多保留 max_per_person 条记录
    3. 然后在 train/test 内部分别按 label 分层抽样：
        - 训练集：每个 label 最多 train_per_class 条
        - 测试集：每个 label 最多 test_per_class 条
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[user_id_col] = df[user_id_col].astype(str)

    # === 第一步：按人划分 train/test ===
    persons = df[user_id_col].dropna().unique().tolist()
    rng.shuffle(persons)
    n = len(persons)
    cut = int(round(n * train_person_ratio))
    cut = max(min(cut, n - 1), 1)   # 确保 train/test 都不为空

    train_ids = set(persons[:cut])
    test_ids = set(persons[cut:])

    train_raw = df[df[user_id_col].isin(train_ids)].copy()
    test_raw = df[df[user_id_col].isin(test_ids)].copy()

    # === 第二步：每人最多 max_per_person 条 ===
    def limit_per_person(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        return (
            d.groupby(user_id_col, group_keys=False)
             .apply(lambda g: g.sample(n=min(len(g), max_per_person), random_state=seed))
             .reset_index(drop=True)
        )

    train_limited = limit_per_person(train_raw)
    test_limited = limit_per_person(test_raw)

    # === 第三步：按 label 分层抽样 ===
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

    train_final = stratified_sample(train_limited, train_per_class)
    test_final = stratified_sample(test_limited, test_per_class)

    print("=== 划分结果统计 ===")
    print("训练集人数:", train_final[user_id_col].nunique(), "样本数:", len(train_final))
    print("训练集各类样本数:\n", train_final[label_col].value_counts().sort_index())
    print("测试集人数:", test_final[user_id_col].nunique(), "样本数:", len(test_final))
    print("测试集各类样本数:\n", test_final[label_col].value_counts().sort_index())

    return train_final, test_final


# ========== 3) 每人前两天→训练，最后一天→测试；每天最多 N 条；控制总样本 ==========
def split_by_day_sample_limit(df: pd.DataFrame,
                              train_sample_limit: int = 1000,
                              test_sample_limit: int = 200,
                              max_per_day: int = 3,
                              random_state: int = 42,
                              verbose: bool = True):
    np.random.seed(random_state)
    df = _prepare_group_and_time(df)

    train_list, test_list, extra_list = [], [], []

    for _, person_df in df.groupby("group_id"):
        person_df = person_df.dropna(subset=["date_obj"]).sort_values("date_obj")
        days = person_df["date_obj"].drop_duplicates().tolist()
        if len(days) == 0:
            continue
        elif len(days) == 1:
            train_list.append(person_df)
        elif len(days) == 2:
            train_list.append(person_df[person_df["date_obj"] == days[0]])
            test_list.append(person_df[person_df["date_obj"] == days[1]])
        else:
            first_two, last_day = days[:2], days[-1]
            middle_days = days[2:-1]
            train_list.append(person_df[person_df["date_obj"].isin(first_two)])
            test_list.append(person_df[person_df["date_obj"] == last_day])
            if middle_days:
                extra = person_df[person_df["date_obj"].isin(middle_days)]
                if not extra.empty:
                    extra_list.append(extra)

    train_df = pd.concat(train_list, ignore_index=True) if train_list else df.iloc[0:0].copy()
    test_df  = pd.concat(test_list,  ignore_index=True) if test_list  else df.iloc[0:0].copy()
    extra_df = pd.concat(extra_list, ignore_index=True) if extra_list else df.iloc[0:0].copy()

    def _cap_per_day(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty: return d
        if d["date_human"].notna().any():
            return (d.groupby(["group_id", "date_human"], group_keys=False)
                     .apply(lambda g: g.sample(n=min(len(g), max_per_day), random_state=random_state))
                     .reset_index(drop=True))
        else:
            return (d.groupby("group_id", group_keys=False)
                     .apply(lambda g: g.sample(n=min(len(g), max_per_day), random_state=random_state))
                     .reset_index(drop=True))

    train_df = _cap_per_day(train_df)
    test_df  = _cap_per_day(test_df)
    extra_df = _cap_per_day(extra_df)

    # 用 extra 补训练/测试的缺口
    miss_train = train_sample_limit - len(train_df)
    if miss_train > 0 and not extra_df.empty:
        add_train = extra_df.sample(n=min(miss_train, len(extra_df)), random_state=random_state)
        train_df = pd.concat([train_df, add_train], ignore_index=True)
        extra_df = extra_df.drop(index=add_train.index, errors="ignore")

    miss_test = test_sample_limit - len(test_df)
    if miss_test > 0 and not extra_df.empty:
        add_test = extra_df.sample(n=min(miss_test, len(extra_df)), random_state=random_state)
        test_df = pd.concat([test_df, add_test], ignore_index=True)
        extra_df = extra_df.drop(index=add_test.index, errors="ignore")

    # 最终限额
    if len(train_df) < train_sample_limit:
        train_df = resample(train_df, n_samples=train_sample_limit, replace=True, random_state=random_state)
    else:
        train_df = train_df.sample(n=train_sample_limit, random_state=random_state)

    if len(test_df) < test_sample_limit:
        test_df = resample(test_df, n_samples=test_sample_limit, replace=True, random_state=random_state)
    else:
        test_df = test_df.sample(n=test_sample_limit, random_state=random_state)

    if verbose:
        print(f"最终训练集涉及人数：{train_df['group_id'].nunique()} / 最终测试集涉及人数：{test_df['group_id'].nunique()}")
    return train_df, test_df


# ========== 4) “前两天训练、最后一天测试”，并按 label/人天做自定义采样，上限测试条数 ==========
def split_train_test_by_group(df: pd.DataFrame,
                              test_sample_limit: int = 1000,
                              seed: int = 42,
                              verbose: bool = True):
    """
    与你原有语义一致：按人划分天数 -> 训练用前两天，测试用最后一天；若超量则下采样到上限。
    """
    np.random.seed(seed)
    df = _prepare_group_and_time(df)

    train_days, test_days = [], []
    for _, person_df in df.groupby("group_id"):
        person_df = person_df.dropna(subset=["date_obj"]).sort_values("date_obj")
        days = person_df["date_obj"].drop_duplicates().tolist()
        if len(days) == 0:
            continue
        elif len(days) == 1:
            train_days.append(person_df)
        elif len(days) == 2:
            train_days.append(person_df[person_df["date_obj"] == days[0]])
            test_days.append(person_df[person_df["date_obj"] == days[1]])
        else:
            train_days.append(person_df[person_df["date_obj"].isin(days[:2])])
            test_days.append(person_df[person_df["date_obj"] == days[-1]])

    train_df = pd.concat(train_days, ignore_index=True) if train_days else df.iloc[0:0].copy()
    test_df  = pd.concat(test_days,  ignore_index=True) if test_days  else df.iloc[0:0].copy()

    # 测试集上限
    if len(test_df) > test_sample_limit:
        test_df = test_df.sample(n=test_sample_limit, random_state=seed).reset_index(drop=True)

    if verbose:
        print(f"训练集人数：{train_df['group_id'].nunique()}，测试集人数：{test_df['group_id'].nunique()}")
    return train_df, test_df


# ========== 5) 固定选多少训练人/测试人，每天抽 N 组 ==========
def split_by_person_fixed_count(
    df: pd.DataFrame,
    train_person_count: int = 30,
    test_person_count: int = 10,
    groups_per_day: int = 1,
    seed: int = 42,
    verbose: bool = True
):
    np.random.seed(seed)
    df = _prepare_group_and_time(df)

    people = df["group_id"].dropna().astype(str).unique()
    if len(people) < train_person_count + test_person_count:
        raise ValueError("数据中可用的人数不足")  

    np.random.shuffle(people)
    train_people = set(people[:train_person_count])
    test_people  = set(people[train_person_count:train_person_count + test_person_count])

    train_df = df[df["group_id"].isin(train_people)].copy()
    test_df  = df[df["group_id"].isin(test_people)].copy()

    def _sample_per_day(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty: return d
        if d["date_human"].notna().any():
            parts = []
            for (_, _), g in d.groupby(["group_id", "date_human"]):
                avail = g["group_time"].drop_duplicates()
                k = min(groups_per_day, len(avail))
                chosen = np.random.choice(avail, size=k, replace=False)
                parts.append(g[g["group_time"].isin(chosen)])
            return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0].copy()
        else:
            return (d.groupby("group_id", group_keys=False)
                     .apply(lambda g: g.sample(n=min(len(g), groups_per_day), random_state=seed))
                     .reset_index(drop=True))

    train_df = _sample_per_day(train_df)
    test_df  = _sample_per_day(test_df)

    if verbose:
        print(f"训练集人数：{train_df['group_id'].nunique()}，样本数：{len(train_df)}")
        print(f"测试集人数：{test_df['group_id'].nunique()}，样本数：{len(test_df)}")
    return train_df, test_df


# ========== 6) TopN 数据量的人做测试；剩余里选 M 人做训练；每天抽 N 组 ==========
def split_by_topN_people(df,
                         test_person_count=5,
                         train_person_count=30,
                         groups_per_day=2,
                         seed=42,
                         verbose=True):
    np.random.seed(seed)
    df = _prepare_group_and_time(df)

    # 统计每人“组数”（若没有真实 time，就用占位 group_time 计数）
    person_group_counts = df.groupby("group_id")["group_time"].nunique().sort_values(ascending=False)
    all_people = person_group_counts.index.tolist()

    test_persons = all_people[:min(test_person_count, len(all_people))]
    remaining = [p for p in all_people if p not in test_persons]
    train_persons = remaining[:min(train_person_count, len(remaining))]

    test_df  = df[df["group_id"].isin(test_persons)].copy()
    train_df = df[df["group_id"].isin(train_persons)].copy()

    def _sample_per_day(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty: return d
        if d["date_human"].notna().any():
            parts = []
            for (_, _), g in d.groupby(["group_id", "date_human"]):
                avail = g["group_time"].drop_duplicates()
                k = min(groups_per_day, len(avail))
                chosen = np.random.choice(avail, size=k, replace=False)
                parts.append(g[g["group_time"].isin(chosen)])
            return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0].copy()
        else:
            return (d.groupby("group_id", group_keys=False)
                     .apply(lambda g: g.sample(n=min(len(g), groups_per_day), random_state=seed))
                     .reset_index(drop=True))

    train_df = _sample_per_day(train_df)
    test_df  = _sample_per_day(test_df)

    if verbose:
        print(f"训练集人数：{train_df['group_id'].nunique()}，样本数：{len(train_df)}")
        print(f"测试集人数：{test_df['group_id'].nunique()}，样本数：{len(test_df)}")
    return train_df, test_df


# ========== 7) “TopN 作测试 + 训练人选策略 + 测试样本上限/分层封顶” ==========
def split_by_person_all_data(df: pd.DataFrame,
                             test_person_count: int = 5,
                             train_person_count: int = 16,
                             volume_metric: str = "rows",    # "rows" 或 "groups"
                             train_select: str = "random",    # "random" 或 "top"
                             seed: int = 42,
                             verbose: bool = True,
                             max_test_samples: int | None = None,
                             cap_strategy: str = "by_person",
                             stratify_col: str = "label"):
    np.random.seed(seed)
    df = _prepare_group_and_time(df)

    # 计算每人数据量
    if volume_metric == "rows":
        person_counts = df.groupby("group_id").size().sort_values(ascending=False)
    elif volume_metric == "groups":
        person_counts = df.groupby("group_id")["group_time"].nunique().sort_values(ascending=False)
    else:
        raise ValueError("volume_metric 只能是 'rows' 或 'groups'")

    all_people = person_counts.index.astype(str).tolist()
    test_people = all_people[:min(test_person_count, len(all_people))]
    remaining = [p for p in all_people if p not in test_people]

    if train_select == "top":
        train_people = remaining[:min(train_person_count, len(remaining))]
    elif train_select == "random":
        k = min(train_person_count, len(remaining))
        train_people = (np.random.choice(remaining, size=k, replace=False).tolist() if k > 0 else [])
    else:
        raise ValueError("train_select 只能是 'random' 或 'top'")

    train_df = df[df["group_id"].astype(str).isin(train_people)].copy()
    test_df  = df[df["group_id"].astype(str).isin(test_people)].copy()

    # 测试集封顶
    if max_test_samples is not None and len(test_df) > max_test_samples:
        if cap_strategy == "by_person":
            per = test_df.groupby("group_id").size().sort_values(ascending=False)
            keep_ids, total = [], 0
            for pid, cnt in per.items():
                cnt = int(cnt)
                if total + cnt > max_test_samples:
                    break
                keep_ids.append(str(pid))
                total += cnt
            test_df = test_df[test_df["group_id"].astype(str).isin(keep_ids)].copy()
            if verbose:
                print(f"[cap/by_person] 测试集人数→{len(keep_ids)}，样本数→{len(test_df)}")
        elif cap_strategy == "stratify":
            if stratify_col in test_df.columns:
                vc = test_df[stratify_col].value_counts()
                quotas_float = vc / vc.sum() * max_test_samples
                quotas = quotas_float.round().astype(int)
                diff = max_test_samples - quotas.sum()
                if diff != 0:
                    order = (quotas_float - quotas).abs().sort_values(ascending=False).index.tolist()
                    for i in range(abs(diff)):
                        quotas.loc[order[i % len(order)]] += 1 if diff > 0 else -1
                parts = []
                for lab, q in quotas.items():
                    part = test_df[test_df[stratify_col] == lab]
                    q = min(q, len(part))
                    parts.append(part.sample(n=q, random_state=seed))
                test_df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
            else:
                test_df = test_df.sample(n=max_test_samples, random_state=seed).reset_index(drop=True)
        else:
            raise ValueError("cap_strategy 只能是 'by_person' 或 'stratify'")

    if verbose:
        print(f"[split_by_person_all_data] 训练集人数：{train_df['group_id'].nunique()}，测试集人数：{test_df['group_id'].nunique()}")

    return train_df, test_df
