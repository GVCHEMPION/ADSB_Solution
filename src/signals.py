from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import topk

W_SAME_LOCATION = 0.4
V_LOG_DISTANCE = 0.01
EARTH_DIAMETER_KM = 12742.0
UNKNOWN_DISTANCE_KM = 3000.0
W_MICROCAT = 0.04
KNN_QUERIES = 20
PRIOR_FLOOR = 1e-4

Adjust = Callable[[slice, npt.NDArray[np.float32]], npt.NDArray[np.float32]]


def location_centroids(*frames: pd.DataFrame) -> pd.DataFrame:
    cols = ["item_location_id", "item_latitude", "item_longitude"]
    items = pd.concat([f[cols] for f in frames])
    return items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()


class Geo(NamedTuple):
    qloc: npt.NDArray[np.int64]
    qlat: npt.NDArray[np.float64]
    qlon: npt.NDArray[np.float64]
    iloc: npt.NDArray[np.int64]
    ilat: npt.NDArray[np.float64]
    ilon: npt.NDArray[np.float64]


def geo_arrays(queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame) -> Geo:
    cent = location_centroids(corpus, rest.drop_duplicates("item_id"))
    loc = queries["search_location_id"]
    return Geo(
        loc.to_numpy(),
        np.radians(loc.map(cent["item_latitude"]).to_numpy(dtype=np.float64)),
        np.radians(loc.map(cent["item_longitude"]).to_numpy(dtype=np.float64)),
        corpus["item_location_id"].to_numpy(),
        np.radians(corpus["item_latitude"].to_numpy(dtype=np.float64)),
        np.radians(corpus["item_longitude"].to_numpy(dtype=np.float64)),
    )


def distance_km(g: Geo, qi: npt.NDArray[np.integer], ci: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
    la, lo, ila, ilo = g.qlat[qi], g.qlon[qi], g.ilat[ci], g.ilon[ci]
    h = np.sin((ila - la) / 2) ** 2 + np.cos(la) * np.cos(ila) * np.sin((ilo - lo) / 2) ** 2
    km: npt.NDArray[np.float64] = np.nan_to_num(
        EARTH_DIAMETER_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1))), nan=UNKNOWN_DISTANCE_KM
    )
    return km


def geo_boost(queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame) -> Adjust:
    g = geo_arrays(queries, corpus, rest)
    all_items = np.arange(len(corpus))[None, :]

    def adjust(rows: slice, scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        qi = np.arange(rows.start, rows.stop)[:, None]
        same = g.qloc[qi] == g.iloc[all_items]
        boost: npt.NDArray[np.float64] = W_SAME_LOCATION * same - V_LOG_DISTANCE * np.log1p(
            distance_km(g, qi, all_items)
        )
        return scores + boost.astype(np.float32)

    return adjust


def normalize_query(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace("ё", "е", regex=False)


def microcat_prior(
    queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
    texts, text_idx = np.unique(normalize_query(rest["search_query"]).to_numpy(), return_inverse=True)
    mcs, mc_idx = np.unique(rest["item_microcat_id"].to_numpy(), return_inverse=True)
    counts = sp.csr_matrix((np.ones(len(rest), np.float32), (text_idx, mc_idx)), shape=(len(texts), len(mcs)))
    dist = sp.csr_matrix(sp.diags(1 / np.asarray(counts.sum(1)).ravel()) @ counts)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float32
    )
    TT = vec.fit_transform(texts).T.tocsr()
    Q = vec.transform(normalize_query(queries["search_query"]))
    prior = np.zeros((len(queries), len(mcs) + 1), np.float32)
    for i in range(0, len(queries), 512):
        idx, sims = topk((Q[i : i + 512] @ TT).toarray(), KNN_QUERIES)
        for r, (nb, w) in enumerate(zip(idx, sims**2, strict=True)):
            prior[i + r, :-1] = w @ dist[nb].toarray()
    prior /= prior.sum(1, keepdims=True) + 1e-9

    col = {m: j for j, m in enumerate(mcs)}
    item_col = corpus["item_microcat_id"].map(col).fillna(len(mcs)).to_numpy(dtype=np.intp)
    return np.log(prior + PRIOR_FLOOR).astype(np.float32), item_col


def microcat_log_prior(queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame) -> Adjust:
    log_prior, item_col = microcat_prior(queries, corpus, rest)

    def adjust(rows: slice, scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return (scores + W_MICROCAT * log_prior[rows][:, item_col]).astype(np.float32, copy=False)

    return adjust


def geo_microcat_boost(queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame) -> Adjust:
    geo = geo_boost(queries, corpus, rest)
    microcat = microcat_log_prior(queries, corpus, rest)
    return lambda rows, scores: microcat(rows, geo(rows, scores))
