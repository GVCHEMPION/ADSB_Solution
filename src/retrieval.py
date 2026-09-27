from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from eval import recall_at_k

K = 1000
CHUNK = 256
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results.tsv"

Candidates = tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]
Part = tuple[sp.csr_matrix, sp.csr_matrix, float, bool]


def topk(scores: npt.NDArray[np.float32], k: int) -> Candidates:
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    s = np.take_along_axis(scores, top, 1)
    order = np.argsort(-s, 1)
    return np.take_along_axis(top, order, 1).astype(np.int32), np.take_along_axis(s, order, 1)


def chunk_scores(m: sp.csr_matrix, max_norm: bool) -> npt.NDArray[np.float32]:
    scores: npt.NDArray[np.float32] = m.toarray()
    if max_norm:
        scores /= scores.max(1, keepdims=True) + 1e-9
    return scores


def sparse_search(
    parts: Sequence[Part],
    k: int = K,
    adjust: Callable[[slice, npt.NDArray[np.float32]], npt.NDArray[np.float32]] | None = None,
) -> Candidates:
    transposed = [(Q, X.T.tocsr(), weight, max_norm) for Q, X, weight, max_norm in parts]
    n_queries, n_items = parts[0][0].shape[0], parts[0][1].shape[0]
    out = []
    for i in tqdm(range(0, n_queries, CHUNK), desc="search"):
        rows = slice(i, min(i + CHUNK, n_queries))
        scores = np.zeros((rows.stop - rows.start, n_items), np.float32)
        for Q, XT, weight, max_norm in transposed:
            scores += weight * chunk_scores(Q[rows] @ XT, max_norm)
        out.append(topk(scores if adjust is None else adjust(rows, scores), k))
    return np.vstack([c[0] for c in out]), np.vstack([c[1] for c in out])


def save(name: str, split: str, cand: Candidates) -> None:
    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / f"{name}_{split}.npz", idx=cand[0], scores=cand[1])


def load(name: str, split: str) -> Candidates | None:
    path = CACHE / f"{name}_{split}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["idx"], data["scores"]


def report(name: str, idx: npt.NDArray[np.int32], queries: pd.DataFrame, corpus: pd.DataFrame) -> None:
    ids = corpus["item_id"].to_numpy()
    rows = []
    for fold in (0, 1):
        mask = (queries["fold"] == fold).to_numpy()
        preds = ids[idx[mask]].tolist()
        relevant = queries.loc[mask, "relevant"].tolist()
        scores = {k: recall_at_k(preds, relevant, k) for k in (50, 200, 1000)}
        print(f"{name:28s} fold{fold}  " + "  ".join(f"@{k} {s:.4f}" for k, s in scores.items()))
        rows.append({"method": name, "fold": fold, **{f"r{k}": round(s, 4) for k, s in scores.items()}})
    old = pd.read_csv(RESULTS, sep="\t") if RESULTS.exists() else pd.DataFrame()
    kept = old[old["method"] != name] if len(old) else old
    pd.concat([kept, pd.DataFrame(rows)]).to_csv(RESULTS, sep="\t", index=False, float_format="%.4f")
