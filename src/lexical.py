import re
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lemmas import lemmatize, lemmatized_items
from retrieval import Candidates, search
from signals import Adjust, geo_boost, geo_microcat_boost

PARAMS_HEAD = 200
BM25_K1 = 1.2
BM25_B = 0.75
W_BM25 = 0.2

USEFUL_KEYS = [
    "Вид услуги",
    "Тип услуги",
    "Услуга",
    "Название услуги",
    "Специальность или сфера",
    "Специальность",
    "Чем вы занимаетесь",
]
NOISE_KEYS = [
    "Место оказания услуг",
    "Начальная цена",
    "Тип стоимости",
    "Опыт работы",
    "Стоимость",
    "График работы",
    "Время работы",
    "Куда выезжаете",
    "Работа по договору",
    "Работаете с юрлицами",
    "Как вы работаете",
    "Продолжительность",
    "Гарантия",
    "Своя услуга",
    "Бригада",
    "Готов закупить материалы",
    "Выполняю заказы от",
    "Время для связи",
    "Предоплата",
    "Рабочие дни",
    "Берёте ли срочные заказы",
    "Дни",
    "Признак предзаполнения прайс листа",
    "Кто оказывает услуги",
    "Где вы оказываете услуги",
    "Ваши клиенты",
    "Год окончания",
]
KEY_RE = re.compile(
    r"(?:^|\s)("
    + "|".join(map(re.escape, sorted(USEFUL_KEYS + NOISE_KEYS, key=len, reverse=True)))
    + r")(?=\s|$)"
)
USEFUL = set(USEFUL_KEYS)


def normalize(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace("ё", "е", regex=False)


def service_fields(params: str) -> str:
    parts = KEY_RE.split(params)
    values = [v.strip() for k, v in zip(parts[1::2], parts[2::2], strict=True) if k in USEFUL]
    return " ".join(dict.fromkeys(v for v in values if v))


def service_kind(params: str | None) -> str:
    parts = KEY_RE.split(params or "")
    return next((v.strip() for k, v in zip(parts[1::2], parts[2::2], strict=True) if k == "Вид услуги"), "")


def text_head(items: pd.DataFrame) -> pd.Series:
    params = items["item_infm_params_text"].fillna("").str[:PARAMS_HEAD]
    return normalize(items["item_title_raw"] + " " + params)


def text_fields(items: pd.DataFrame) -> pd.Series:
    fields = items["item_infm_params_text"].fillna("").map(service_fields)
    return normalize(items["item_title_raw"] + " " + fields)


def char_matrices(
    queries: pd.DataFrame, corpus: pd.DataFrame, text: Callable[[pd.DataFrame], pd.Series]
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, dtype=np.float32
    )
    X = vec.fit_transform(text(corpus))
    return vec.transform(normalize(queries["search_query"])), X


def char_tfidf(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    rest: pd.DataFrame,
    text: Callable[[pd.DataFrame], pd.Series] = text_head,
    boost: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], Adjust] | None = None,
) -> Candidates:
    Q, X = char_matrices(queries, corpus, text)
    return search([(Q, X, 1.0, False)], adjust=boost(queries, corpus, rest) if boost else None)


char_tfidf_fields = partial(char_tfidf, text=text_fields)
char_tfidf_geo = partial(char_tfidf, boost=geo_boost)
char_tfidf_geo_mc = partial(char_tfidf, boost=geo_microcat_boost)


def text_with_description(items: pd.DataFrame) -> pd.Series:
    return text_head(items) + " " + normalize(items["item_description"])


def bm25_weights(counts: sp.csr_matrix) -> sp.csr_matrix:
    lengths = np.asarray(counts.sum(1)).ravel()
    df = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((counts.shape[0] - df + 0.5) / (df + 0.5))
    rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    tf = counts.data
    norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths[rows] / lengths.mean())
    data = idf[counts.indices] * tf * (BM25_K1 + 1) / (tf + norm)
    return sp.csr_matrix((data.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape)


def bm25_matrices(
    queries: pd.DataFrame, corpus: pd.DataFrame, text: Callable[[pd.DataFrame], pd.Series]
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    vec = CountVectorizer(analyzer=str.split, dtype=np.float32)
    W = bm25_weights(
        vec.fit_transform(lemmatized_items(corpus["item_id"], text(corpus), text.__name__)).tocsr()
    )
    Q = vec.transform(lemmatize(queries["search_query"])).sign()
    return Q, W


def hybrid_geo_mc(queries: pd.DataFrame, corpus: pd.DataFrame, rest: pd.DataFrame) -> Candidates:
    Q, X = char_matrices(queries, corpus, text_head)
    Qb, W = bm25_matrices(queries, corpus, text_with_description)
    return search(
        [(Q, X, 1.0, False), (Qb, W, W_BM25, True)], adjust=geo_microcat_boost(queries, corpus, rest)
    )
