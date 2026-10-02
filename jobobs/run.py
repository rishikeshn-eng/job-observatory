from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import audits, dedupe, synth


def pair_scores(found: set, truth: set):
    tp = len(found & truth)
    return {"precision": tp / max(1, len(found)), "recall": tp / max(1, len(truth)), "found": len(found), "truth": len(truth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="postings CSV in the Kaggle 2023-24 schema (real data)")
    ap.add_argument("--roles", type=int, default=9000)
    ap.add_argument("--out", default="docs/results.json")
    a = ap.parse_args()
    truth = None
    if a.csv:
        df = pd.read_csv(a.csv, parse_dates=["listed_time", "original_listed_time", "expiry"])
        if "state" not in df:
            df["state"] = df.location.str.extract(r",\s*([A-Z]{2})\b")[0]
        synthetic = False
    else:
        df, truth = synth.generate(a.roles)
        synthetic = True
    df = df.reset_index(drop=True)
    idx = {j: i for i, j in enumerate(df.job_id)}
    texts = df.description.tolist()

    dedupe_res = {}
    for name, mdf in (("with boilerplate filter", 0.05), ("no boilerplate filter", None)):
        pairs, ncand = dedupe.find_duplicates(texts, threshold=0.6, max_df=mdf)
        cross = {p for p in pairs if df.company_name[p[0]] != df.company_name[p[1]]}
        row = {"verified_pairs": len(pairs), "cross_company_pairs": len(cross), "candidates": ncand}
        if truth:
            t = {tuple(sorted((idx[a_], idx[b_]))) for a_, b_ in truth["dup_pairs"]}
            row.update(pair_scores({tuple(sorted(p)) for p in cross}, t))
        dedupe_res[name] = row
        if mdf:
            good_pairs = pairs
        print(name, row)
    cl = dedupe.clusters(len(df), good_pairs)
    cross_clusters = [g for g in cl if df.company_name.iloc[g].nunique() > 1]
    cross_clusters.sort(key=lambda g: -df.company_name.iloc[g].nunique())
    examples = [{"size": len(g), "companies": sorted(df.company_name.iloc[g].unique())[:6], "n_companies": int(df.company_name.iloc[g].nunique()),
                 "title": df.title.iloc[g[0]], "text": df.description.iloc[g[0]][:220]} for g in cross_clusters[:8]]

    rp = audits.reposts(df)
    if truth:  # hidden reposts: how many of the planted chain links without a platform link does text-dedupe recover?
        chain_pairs = {tuple(sorted((idx[c[i]], idx[c[j]]))) for c in truth["repost_chains"] for i in range(len(c)) for j in range(i + 1, len(c))}
        same_co = {tuple(sorted(p)) for p in good_pairs if df.company_name[p[0]] == df.company_name[p[1]]}
        rp["dedupe_recovers_chain_pairs"] = pair_scores(same_co, chain_pairs)
    pay = audits.pay(df)
    w = audits.wording(df)
    w_raw = audits.wording(df, strip_title=False)
    fam = (w.groupby("family").agg(postings=("job_id", "count"), masc=("masc", "mean"), fem=("fem", "mean"), skew=("gskew", "mean"))
           .join(w_raw.groupby("family").gskew.mean().rename("skew_incl_title_words"))
           .sort_values("skew", ascending=False).round(3).reset_index().to_dict("records"))
    wording = {"mean_skew": float(w.gskew.mean()), "share_masc_coded": float((w.gskew > 0.25).mean()),
               "share_fem_coded": float((w.gskew < -0.25).mean()), "families": fam}
    payload = {"synthetic": synthetic, "postings": int(len(df)), "companies": int(df.company_name.nunique()),
               "dedupe": dedupe_res, "dup_examples": examples, "reposts": rp, "pay": pay, "wording": wording}
    Path(a.out).write_text(json.dumps(payload, default=float))
    print("relisted share", round(rp["relisted_share"], 3), "| pay law-state", round(pay["disclosed_law_states"], 3),
          "other", round(pay["disclosed_other_states"], 3))


if __name__ == "__main__":
    main()
