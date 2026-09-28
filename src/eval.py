import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "dataset"

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = ["item_id", "item_title_raw", "item_infm_params_text"]
ITEM_META = [
    "item_category_id",
    "item_microcat_id",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
]

CORPUS_SIZE = 189_212
DESCRIPTION_HEAD = 500
ID_RE = re.compile(r"^[0-9a-f]{16}$")


def recall_at_k(predictions: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int = 50) -> float:
    scores = [len(set(p[:k]) & r) / len(r) for p, r in zip(predictions, relevant, strict=True)]
    return float(np.mean(scores))


def read(name: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(DATA / name, columns=columns)
    for col in ("item_latitude", "item_longitude", "item_price"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def read_descriptions(name: str, ids: pd.Series) -> pd.Series:
    table = pq.read_table(DATA / name, columns=["item_id", "item_description_raw"])
    head = pc.utf8_slice_codeunits(pc.fill_null(table["item_description_raw"], ""), 0, DESCRIPTION_HEAD)
    desc = pd.Series(
        head.to_numpy(zero_copy_only=False), index=table["item_id"].to_numpy(zero_copy_only=False)
    )
    desc = desc[~desc.index.duplicated()]
    return pd.Series(desc.reindex(ids.to_numpy()).fillna("").to_numpy(), index=ids.index)


def make_holdout(n_queries: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = read("train.parquet", QUERY_COLS + ITEM_COLS + ITEM_META)
    group_id = tr.groupby(QUERY_COLS, dropna=False, sort=False).ngroup()

    rng = np.random.default_rng(seed)
    first = pd.Series(group_id.to_numpy()).groupby(tr["search_query"].to_numpy()).first()
    held = set(rng.choice(first.to_numpy(), size=n_queries, replace=False).tolist())
    mask = group_id.isin(held).to_numpy()

    queries = (
        tr[mask]
        .groupby(QUERY_COLS, dropna=False, sort=False)["item_id"]
        .agg(set)
        .rename("relevant")
        .reset_index()
    )
    queries["fold"] = rng.permutation(len(queries)) % 2

    items = tr[ITEM_COLS + ITEM_META].drop_duplicates("item_id")
    must: set[str] = set().union(*queries["relevant"])
    is_must = items["item_id"].isin(must)
    fill = items[~is_must].sample(n=CORPUS_SIZE - len(must), random_state=seed)
    corpus = pd.concat([items[is_must], fill], ignore_index=True)
    corpus["item_description"] = read_descriptions("train.parquet", corpus["item_id"])

    return queries, corpus, tr[~mask]


def load_bench() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    items = read("benchmark_items.parquet", ITEM_COLS + ITEM_META)
    items["item_description"] = read_descriptions("benchmark_items.parquet", items["item_id"])
    train = read("train.parquet", QUERY_COLS + ITEM_COLS + ITEM_META)
    return queries, items, train


def check_answer(path: str) -> None:
    header = Path(path).read_text(encoding="utf-8").splitlines()[0]
    assert header == "query_id,answer", f"заголовок должен быть 'query_id,answer', а он {header!r}"

    ans = pd.read_csv(path, dtype=str, keep_default_na=False)
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet", columns=["query_id"])
    corpus = set(pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])["item_id"])

    assert list(ans.columns) == ["query_id", "answer"], f"лишние колонки: {list(ans.columns)}"
    assert ans["query_id"].is_unique, "повторяющиеся query_id"
    missing = set(bq["query_id"]) - set(ans["query_id"])
    extra = set(ans["query_id"]) - set(bq["query_id"])
    assert not missing, f"нет ответа для {len(missing)} query_id"
    assert not extra, f"{len(extra)} лишних query_id"

    sizes: list[int] = []
    for qid, a in zip(ans["query_id"], ans["answer"], strict=True):
        ids = a.split()
        sizes.append(len(ids))
        assert len(ids) <= 50, f"{qid}: {len(ids)} id больше 50"
        assert len(ids) == len(set(ids)), f"{qid}: повторы внутри строки"
        bad = [i for i in ids if not ID_RE.match(i) or i not in corpus]
        assert not bad, f"{qid}: id не из корпуса или кривого формата: {bad[:3]}"

    mean = sum(sizes) / len(sizes)
    print(f"OK: {len(ans)} строк, id на строку: min {min(sizes)}, mean {mean:.1f}, max {max(sizes)}")


if __name__ == "__main__":
    check_answer(sys.argv[1] if len(sys.argv) > 1 else "answer.csv")
