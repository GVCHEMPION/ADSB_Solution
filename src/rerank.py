import argparse
from functools import cache

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

import retrieval
from eval import load_bench, make_holdout, recall_at_k

RANKER = "ranker_bgeft_hybrid_bgeft"
MODELS = {"bge": "bge-reranker-v2-m3"}
DEPTH = 100
MAX_LEN = 256
BATCH = 64
PARAMS_HEAD = 200
DESCRIPTION_HEAD = 300

Scores = npt.NDArray[np.float32]


def item_text(items: pd.DataFrame) -> pd.Series:
    params = items["item_infm_params_text"].fillna("").str[:PARAMS_HEAD]
    description = items["item_description"].fillna("").str[:DESCRIPTION_HEAD]
    return items["item_title_raw"] + ". " + params + ". " + description


@cache
def model(name: str) -> CrossEncoder:
    ce: CrossEncoder = CrossEncoder(
        str(retrieval.ROOT / "models" / MODELS[name]),
        device="cuda",
        max_length=MAX_LEN,
        model_kwargs={"torch_dtype": torch.float16},
    )
    return ce


def cross_scores(
    name: str, split: str, queries: pd.DataFrame, corpus: pd.DataFrame, cand: npt.NDArray[np.int32]
) -> Scores:
    path = retrieval.CACHE / f"ce_{name}_{split}_{cand.shape[1]}.npy"
    if path.exists():
        cached: Scores = np.load(path)
        return cached
    texts = item_text(corpus).to_numpy()
    pairs = [(q, texts[c]) for q, row in zip(queries["search_query"], cand, strict=True) for c in row]
    with torch.inference_mode():
        flat = model(name).predict(pairs, batch_size=BATCH, show_progress_bar=True, convert_to_numpy=True)
    scores = np.asarray(flat, dtype=np.float32).reshape(cand.shape)
    np.save(path, scores)
    return scores


def zscore(s: Scores) -> Scores:
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def blend(cand: npt.NDArray[np.int32], ranker: Scores, cross: Scores, w: float) -> npt.NDArray[np.int32]:
    depth = cross.shape[1]
    mixed = zscore(ranker[:, :depth]) + w * zscore(cross)
    order = np.argsort(-mixed, 1)
    top = np.take_along_axis(cand[:, :depth], order, 1)
    return np.hstack([top, cand[:, depth:]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge", choices=sorted(MODELS))
    ap.add_argument(
        "--depth", type=int, default=DEPTH, help="сколько верхних кандидатов ранкера переоценивать"
    )
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--bench", type=float, default=None, help="вес reranker для боя; пишет answer.csv")
    ap.add_argument("--out", default="answer.csv")
    args = ap.parse_args()

    if args.bench is None:
        queries, corpus, _ = make_holdout()
        cached = retrieval.load(RANKER, "valid")
        assert cached is not None, "сначала прогони ranker.py"
        cand, ranker_scores = cached
        cross = cross_scores(args.model, "valid", queries, corpus, cand[:, : args.depth])
        ids = corpus["item_id"].to_numpy()
        fold = queries["fold"].to_numpy()
        for w in args.weights:
            top = ids[blend(cand, ranker_scores, cross, w)[:, :50]]
            r = [
                recall_at_k(top[fold == f].tolist(), queries.loc[fold == f, "relevant"].tolist())
                for f in (0, 1)
            ]
            print(f"{args.model} depth={args.depth} w={w:<5} fold0 {r[0]:.4f}  fold1 {r[1]:.4f}")
        return

    bq, items, _ = load_bench()
    cached = retrieval.load(RANKER, "bench")
    assert cached is not None, "сначала прогони ranker.py --bench"
    cand, ranker_scores = cached
    cross = cross_scores(args.model, "bench", bq, items, cand[:, : args.depth])
    top = items["item_id"].to_numpy()[blend(cand, ranker_scores, cross, args.bench)[:, :50]]
    answer = pd.DataFrame({"query_id": bq["query_id"], "answer": [" ".join(row) for row in top]})
    answer.to_csv(args.out, index=False)
    print(f"записано {args.out}: {len(answer)} запросов")


if __name__ == "__main__":
    main()
