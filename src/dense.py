import argparse
from functools import cache

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

from eval import make_holdout
from lexical import W_BM25, bm25_matrices, char_matrices, normalize, text_head, text_with_description
from retrieval import CACHE, ROOT, Candidates, Part, search
from signals import geo_microcat_boost

MODELS = {
    "e5": ("multilingual-e5-base", "query: ", "passage: "),
    "bge": ("USER-bge-m3", "", ""),
    "e5ft": ("e5-ft", "query: ", "passage: "),
    "bgeft": ("bge-ft", "", ""),
}
BASE = {"e5ft": "e5", "bgeft": "bge"}
MAX_LEN = 128
W_LEXICAL = 0.2
BATCH = 256

Embeddings = npt.NDArray[np.float16]


@cache
def model(name: str) -> SentenceTransformer:
    st = SentenceTransformer(str(ROOT / "models" / MODELS[name][0]), device="cuda")
    st.half()
    st.max_seq_length = MAX_LEN
    return st


def encode(name: str, texts: list[str], query: bool) -> Embeddings:
    prefix = MODELS[name][1] if query else MODELS[name][2]
    with torch.inference_mode():
        emb = model(name).encode(
            [prefix + t for t in texts],
            batch_size=BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > BATCH,
        )
    return emb.astype(np.float16)


def item_embeddings(name: str, ids: pd.Series, texts: pd.Series, variant: str) -> Embeddings:
    path = CACHE / f"emb_{name}_{variant}.npz"
    known_ids, known = (np.array([], dtype=object), None)
    if path.exists():
        data = np.load(path, allow_pickle=True)
        known_ids, known = data["ids"], data["emb"]
    pos = pd.Series(np.arange(len(known_ids)), index=known_ids)
    missing = ~ids.isin(pos.index)
    if missing.any():
        fresh = encode(name, texts[missing].tolist(), query=False)
        known_ids = np.concatenate([known_ids, ids[missing].to_numpy(dtype=object)])
        known = fresh if known is None else np.vstack([known, fresh])
        CACHE.mkdir(exist_ok=True)
        np.savez(path, ids=known_ids, emb=known)
        pos = pd.Series(np.arange(len(known_ids)), index=known_ids)
    assert known is not None
    emb: Embeddings = known[pos.loc[ids.to_numpy()].to_numpy()]
    return emb


def dense_hybrid(queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame, name: str) -> Candidates:
    D = item_embeddings(name, corpus["item_id"], text_head(corpus), "text_head")
    Qe = encode(name, queries["search_query"].tolist(), query=True)
    Q, X = char_matrices(queries, corpus, text_head)
    Qb, W = bm25_matrices(queries, corpus, text_with_description)
    parts: list[Part] = [(Qe, D, 1.0, False), (Q, X, W_LEXICAL, False), (Qb, W, W_LEXICAL * W_BM25, True)]
    return search(parts, adjust=geo_microcat_boost(queries, corpus, rest))


def train_pairs(rest: pd.DataFrame, max_pairs: int, seed: int = 0) -> pd.DataFrame:
    pairs = pd.DataFrame(
        {"anchor": normalize(rest["search_query"]), "positive": text_head(rest), "item_id": rest["item_id"]}
    ).drop_duplicates(["anchor", "item_id"])
    return pairs.sample(n=min(max_pairs, len(pairs)), random_state=seed)[["anchor", "positive"]]


def finetune(name: str, max_pairs: int, epochs: int, batch: int, mini_batch: int, lr: float) -> None:
    base = BASE[name]
    st = SentenceTransformer(str(ROOT / "models" / MODELS[base][0]), device="cuda")
    st.max_seq_length = MAX_LEN
    _, _, rest = make_holdout()
    pairs = train_pairs(rest, max_pairs)
    pairs["anchor"] = MODELS[base][1] + pairs["anchor"]
    pairs["positive"] = MODELS[base][2] + pairs["positive"]
    out = ROOT / "models" / MODELS[name][0]
    args = SentenceTransformerTrainingArguments(
        output_dir=str(CACHE / f"train_{name}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(
        model=st,
        args=args,
        train_dataset=Dataset.from_pandas(pairs, preserve_index=False),
        loss=CachedMultipleNegativesRankingLoss(st, mini_batch_size=mini_batch),
    )
    trainer.train()
    st.save(str(out))
    print(f"сохранено в {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(BASE))
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mini-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    finetune(a.name, a.pairs, a.epochs, a.batch, a.mini_batch, a.lr)
