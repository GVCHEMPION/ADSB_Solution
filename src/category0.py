import argparse

import numpy as np
import pandas as pd
import torch

from dense import encode, item_embeddings
from eval import load_bench
from lexical import text_head
from signals import V_LOG_DISTANCE, W_SAME_LOCATION, distance_km, geo_arrays

SLOTS = 10
DENSE = "bgeft"
SERVICES = 114


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="answer.csv")
    ap.add_argument("--out", default="answer.csv")
    ap.add_argument("--slots", type=int, default=SLOTS)
    args = ap.parse_args()

    queries, items, train = load_bench()
    answer = pd.read_csv(args.inp, dtype=str, keep_default_na=False).set_index("query_id")
    lists = answer.loc[queries["query_id"], "answer"].str.split().tolist()

    target = np.flatnonzero((queries["search_category"] == 0).to_numpy())
    others = np.flatnonzero((items["item_category_id"] != SERVICES).to_numpy())
    sub_q = queries.iloc[target].reset_index(drop=True)
    sub_i = items.iloc[others].reset_index(drop=True)

    Qe = torch.from_numpy(encode(DENSE, sub_q["search_query"].tolist(), query=True)).float()
    D = torch.from_numpy(
        item_embeddings(DENSE, items["item_id"], text_head(items), "text_head")[others]
    ).float()
    g = geo_arrays(sub_q, sub_i, train)
    qi = np.arange(len(sub_q))[:, None]
    ci = np.arange(len(sub_i))[None, :]
    geo = W_SAME_LOCATION * (g.qloc[qi] == g.iloc[ci]) - V_LOG_DISTANCE * np.log1p(distance_km(g, qi, ci))
    score = (Qe @ D.T).numpy() + geo
    ids = sub_i["item_id"].to_numpy()

    inserted = 0
    for row, order in zip(target, np.argsort(-score, 1), strict=True):
        keep = lists[row][: 50 - args.slots]
        seen = set(keep)
        extra = [i for i in ids[order] if i not in seen][: args.slots]
        inserted += len(set(extra) - set(lists[row]))
        lists[row] = keep + extra

    out = pd.DataFrame({"query_id": queries["query_id"], "answer": [" ".join(x) for x in lists]})
    out.to_csv(args.out, index=False)
    print(f"записано {args.out}: запросов без категории {len(target)}, новых объявлений {inserted}")


if __name__ == "__main__":
    main()
