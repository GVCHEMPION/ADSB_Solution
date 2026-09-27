import argparse
from collections.abc import Callable
from functools import partial

import pandas as pd

import dense
import lexical
import retrieval
from eval import load_bench, make_holdout

Method = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], retrieval.Candidates]

METHODS: dict[str, Method] = {
    "char_tfidf": lexical.char_tfidf,
    "char_tfidf_fields": lexical.char_tfidf_fields,
    "char_tfidf_geo": lexical.char_tfidf_geo,
    "char_tfidf_geo_mc": lexical.char_tfidf_geo_mc,
    "hybrid_geo_mc": lexical.hybrid_geo_mc,
    "e5ft_hybrid": partial(dense.dense_hybrid, name="e5ft"),
    "bgeft_hybrid": partial(dense.dense_hybrid, name="bgeft"),
}


def run(
    name: str, split: str, queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame, force: bool
) -> retrieval.Candidates:
    cand = None if force else retrieval.load(name, split)
    if cand is None:
        cand = METHODS[name](queries, corpus, rest)
        retrieval.save(name, split, cand)
    return cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=sorted(METHODS))
    ap.add_argument("--bench", action="store_true", help="считать на боевых данных и писать answer.csv")
    ap.add_argument("--force", action="store_true", help="пересчитать, даже если есть кэш")
    ap.add_argument("--out", default="answer.csv")
    args = ap.parse_args()

    if not args.bench:
        queries, corpus, rest = make_holdout()
        cand = run(args.method, "valid", queries, corpus, rest, args.force)
        retrieval.report(args.method, cand[0], queries, corpus)
        return

    queries, corpus, train = load_bench()
    cand = run(args.method, "bench", queries, corpus, train, args.force)
    ids = corpus["item_id"].to_numpy()
    answer = pd.DataFrame(
        {"query_id": queries["query_id"], "answer": [" ".join(row) for row in ids[cand[0][:, :50]]]}
    )
    answer.to_csv(args.out, index=False)
    print(f"записано {args.out}: {len(answer)} запросов")


if __name__ == "__main__":
    main()
