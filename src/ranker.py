import argparse
import gc

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

import llm
import retrieval
from dense import encode, item_embeddings
from eval import load_bench, make_holdout
from lexical import bm25_matrices, char_matrices, service_kind, text_head, text_with_description
from signals import distance_km, geo_arrays, location_affinity, microcat_prior, normalize_query

POOL = 500
BASE = "bgeft_hybrid"
DENSE = ("bgeft",)
Idx = npt.NDArray[np.int32]


def pair_sparse(Q: sp.csr_matrix, X: sp.csr_matrix, cand: Idx) -> npt.NDArray[np.float32]:
    out = np.empty(cand.shape, np.float32)
    for i, row in enumerate(cand):
        out[i] = (X[row] @ Q[i].T).toarray().ravel()
    return out


def pair_dense(Qe: npt.NDArray[np.float16], D: npt.NDArray[np.float16], cand: Idx) -> npt.NDArray[np.float32]:
    out = np.empty(cand.shape, np.float32)
    for i in range(0, len(cand), 256):
        block = D[cand[i : i + 256]].astype(np.float32)
        out[i : i + 256] = np.einsum("qkd,qd->qk", block, Qe[i : i + 256].astype(np.float32))
    return out


def llm_features(
    queries: pd.DataFrame, corpus: pd.DataFrame, cand: Idx, tasks: tuple[str, ...]
) -> dict[str, npt.NDArray[np.float64]]:
    texts = normalize_query(queries["search_query"]).tolist()
    cols: dict[str, npt.NDArray[np.float64]] = {}
    if "kinds" in tasks:
        predicted = llm.answers("kinds")
        kinds = np.array([service_kind(p) for p in corpus["item_infm_params_text"]], dtype=object)
        weight = np.full(cand.shape, np.nan)
        for i, text in enumerate(texts):
            answer = predicted.get(text)
            if isinstance(answer, list):
                rank = {k: 1 / (pos + 1) for pos, k in enumerate(answer)}
                weight[i] = [rank.get(k, 0.0) for k in kinds[cand[i]]]
        print(f"llm kinds: ответ есть у {np.mean([t in predicted for t in texts]):.1%} запросов")
        cols["llm_kind"] = weight
    if "rewrite" in tasks:
        predicted = llm.answers("rewrite")
        rewritten = [str(predicted.get(t, t)) for t in texts]
        print(f"llm rewrite: ответ есть у {np.mean([t in predicted for t in texts]):.1%} запросов")
        D = item_embeddings(DENSE[0], corpus["item_id"], text_head(corpus), "text_head")
        cols["llm_rewrite_dense"] = pair_dense(encode(DENSE[0], rewritten, query=True), D, cand).astype(
            np.float64
        )
    return cols


def features(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    rest: pd.DataFrame,
    cand: Idx,
    scores: npt.NDArray[np.float32],
    dense: tuple[str, ...] = DENSE,
    llm_tasks: tuple[str, ...] = (),
) -> pd.DataFrame:
    Q, X = char_matrices(queries, corpus, text_head)
    Qb, W = bm25_matrices(queries, corpus, text_with_description)
    g = geo_arrays(queries, corpus, rest)
    log_prior, item_col = microcat_prior(queries, corpus, rest)
    affinity, evidence = location_affinity(queries, corpus, rest, cand)
    qi = np.repeat(np.arange(len(queries))[:, None], cand.shape[1], 1)
    bm25 = pair_sparse(Qb, W, cand)

    def item(col: str) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = corpus[col].to_numpy(dtype=np.float64)[cand]
        return values

    def query(values: pd.Series) -> npt.NDArray[np.float64]:
        per_query: npt.NDArray[np.float64] = values.to_numpy(dtype=np.float64)[qi]
        return per_query

    dense_cols = {
        f"dense_{name}": pair_dense(
            encode(name, queries["search_query"].tolist(), query=True),
            item_embeddings(name, corpus["item_id"], text_head(corpus), "text_head"),
            cand,
        )
        for name in dense
    }
    cols = {
        **dense_cols,
        **llm_features(queries, corpus, cand, llm_tasks),
        "char": pair_sparse(Q, X, cand),
        "bm25": bm25,
        "bm25_rel": bm25 / (bm25.max(1, keepdims=True) + 1e-9),
        "same_location": g.qloc[qi] == g.iloc[cand],
        "log_km": np.log1p(distance_km(g, qi, cand)),
        "log_microcat": log_prior[qi, item_col[cand]],
        "location_affinity": affinity,
        "location_evidence": evidence[qi],
        "pool_score": scores,
        "pool_rank": np.broadcast_to(np.arange(cand.shape[1]), cand.shape),
        "item_rating": item("item_rating"),
        "item_reviews": np.log1p(item("item_rating_reviews_count")),
        "item_price": np.log1p(item("item_price")),
        "title_len": corpus["item_title_raw"].str.len().to_numpy()[cand],
        "description_len": corpus["item_description"].str.len().to_numpy()[cand],
        "services_category": item("item_category_id") == 114,
        "query_words": query(queries["search_query"].map(lambda t: len(str(t).split()))),
        "query_category": query(queries["search_category"]),
        "query_has_filters": query(queries["search_infm_params_text"].fillna("").str.len() > 0),
    }
    return pd.DataFrame({name: np.asarray(v, dtype=np.float32).ravel() for name, v in cols.items()})


def labels(queries: pd.DataFrame, corpus: pd.DataFrame, cand: Idx) -> npt.NDArray[np.int32]:
    ids = corpus["item_id"].to_numpy()[cand]
    hits = [np.isin(row, list(rel)) for row, rel in zip(ids, queries["relevant"], strict=True)]
    return np.asarray(hits, dtype=np.int32).ravel()


def fit(X: pd.DataFrame, y: npt.NDArray[np.int32], n_queries: int) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        lambdarank_truncation_level=60,
        verbose=-1,
    )
    model.fit(X, y, group=[POOL] * n_queries)
    return model


def rerank(model: lgb.LGBMRanker, X: pd.DataFrame, cand: Idx) -> retrieval.Candidates:
    pred = np.asarray(model.predict(X), dtype=np.float32).reshape(cand.shape)
    order = np.argsort(-pred, 1)
    return np.take_along_axis(cand, order, 1), np.take_along_axis(pred, order, 1)


def rows_of(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return np.repeat(mask, POOL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="учить на всей валидации и писать answer.csv")
    ap.add_argument("--out", default="answer.csv")
    ap.add_argument("--base", default=BASE, help="метод, чьи кандидаты образуют пул")
    ap.add_argument("--dense", nargs="+", default=list(DENSE), help="dense-модели для признаков")
    ap.add_argument("--llm", nargs="*", default=[], choices=sorted(llm.TASKS), help="признаки из ответов LLM")
    args = ap.parse_args()
    dense, llm_tasks = tuple(args.dense), tuple(args.llm)
    name = f"ranker_{args.base}_{'+'.join(dense)}" + "".join(f"_llm-{t}" for t in llm_tasks)

    queries, corpus, rest = make_holdout()
    cached = retrieval.load(args.base, "valid")
    assert cached is not None, f"сначала посчитай кандидатов: experiment.py {args.base}"
    cand, scores = cached[0][:, :POOL], cached[1][:, :POOL]
    X = features(queries, corpus, rest, cand, scores, dense, llm_tasks)
    y = labels(queries, corpus, cand)
    fold = queries["fold"].to_numpy()

    if not args.bench:
        reranked, reranked_scores = cand.copy(), np.zeros(cand.shape, np.float32)
        for train_fold in (0, 1):
            tr, te = fold == train_fold, fold != train_fold
            model = fit(X[rows_of(tr)], y[rows_of(tr)], int(tr.sum()))
            reranked[te], reranked_scores[te] = rerank(model, X[rows_of(te)], cand[te])
        retrieval.save(name, "valid", (reranked, reranked_scores))
        retrieval.report(name, reranked, queries, corpus)
        importance = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
        print((importance / importance.sum()).sort_values(ascending=False).round(3).to_string())
        return

    model = fit(X, y, len(queries))
    del X, y, queries, corpus, rest, cand, scores
    gc.collect()
    bq, items, train = load_bench()
    cached = retrieval.load(args.base, "bench")
    assert cached is not None, f"сначала посчитай кандидатов: experiment.py {args.base} --bench"
    bcand, bscores = cached[0][:, :POOL], cached[1][:, :POOL]
    ranked = rerank(model, features(bq, items, train, bcand, bscores, dense, llm_tasks), bcand)
    retrieval.save(name, "bench", ranked)
    top = ranked[0][:, :50]
    ids = items["item_id"].to_numpy()
    answer = pd.DataFrame({"query_id": bq["query_id"], "answer": [" ".join(row) for row in ids[top]]})
    answer.to_csv(args.out, index=False)
    print(f"записано {args.out}: {len(answer)} запросов")


if __name__ == "__main__":
    main()
