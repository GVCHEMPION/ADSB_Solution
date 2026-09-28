import re
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import pandas as pd
from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger
from tqdm import tqdm

from retrieval import CACHE

WORD_RE = re.compile(r"\w+")
MAX_TOKENS = 200
GROUP = 64
POOL_CHUNK = 16
WORKERS = 8


@cache
def models() -> tuple[NewsMorphTagger, MorphVocab]:
    return NewsMorphTagger(NewsEmbedding()), MorphVocab()


@cache
def lemma(word: str, pos: str, feats: tuple[tuple[str, str], ...]) -> str:
    result: str = models()[1].lemmatize(word, pos, dict(feats))
    return result


def tag_sequences(sequences: list[list[str]]) -> list[list[str]]:
    tagger, _ = models()
    return [
        [lemma(tok.text, tok.pos, tuple(sorted(tok.feats.items()))) for tok in markup.tokens]
        for markup in tagger.map(sequences)
    ]


def lemmatize(texts: pd.Series) -> pd.Series:
    words = [WORD_RE.findall(t.lower().replace("ё", "е"))[:MAX_TOKENS] for t in texts.fillna("")]
    sequences, spans = [], []
    for i in range(0, len(words), GROUP):
        seq: list[str] = []
        span = []
        for w in words[i : i + GROUP]:
            span.append((len(seq), len(seq) + len(w)))
            seq += [*w, "."]
        sequences.append(seq)
        spans.append(span)

    chunks = [sequences[i : i + POOL_CHUNK] for i in range(0, len(sequences), POOL_CHUNK)]
    if len(chunks) < WORKERS:
        tagged = tag_sequences(sequences)
    else:
        with ProcessPoolExecutor(WORKERS) as pool:
            parts = tqdm(pool.map(tag_sequences, chunks), total=len(chunks), desc="lemmas")
            tagged = [tokens for part in parts for tokens in part]

    out = [" ".join(tokens[a:b]) for tokens, span in zip(tagged, spans, strict=True) for a, b in span]
    return pd.Series(out, index=texts.index)


def lemmatized_items(ids: pd.Series, texts: pd.Series, variant: str) -> pd.Series:
    path = CACHE / f"lemmas_{variant}.parquet"
    known = pd.read_parquet(path)["lemmas"] if path.exists() else pd.Series(dtype=str)
    missing = ~ids.isin(known.index)
    if missing.any():
        fresh = lemmatize(texts[missing])
        fresh.index = ids[missing].to_numpy()
        known = pd.concat([known, fresh])
        known = known[~known.index.duplicated()]
        CACHE.mkdir(exist_ok=True)
        known.rename("lemmas").to_frame().to_parquet(path)
    return pd.Series(known.reindex(ids.to_numpy()).to_numpy(), index=ids.index)
