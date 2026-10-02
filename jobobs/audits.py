"""Repost behaviour, pay-disclosure and gendered-wording audits."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LAW_STATES = {"CA", "NY", "CO", "WA"}  # states with pay-range disclosure rules for postings (illustrative set)

# Masculine- and feminine-coded stems after Gaucher, Friesen & Kay (2011), "Evidence that gendered wording in job
# advertisements exists and sustains gender inequality", JPSP 101(1). A partial transcription; check the paper's
# appendix before relying on completeness.
MASC = ["active", "adventurous", "aggress", "ambitio", "analy", "assert", "athlet", "autonom", "boast", "challeng",
        "compet", "confident", "courag", "decide", "decisive", "decision", "determin", "dominant", "domina", "force",
        "greedy", "headstrong", "hierarch", "hostil", "impulsive", "independen", "individual", "intellect", "lead",
        "logic", "masculine", "objective", "opinion", "outspoken", "persist", "principle", "reckless", "stubborn",
        "superior", "self-confiden", "self-relian", "self-sufficien", "driven", "fearless"]
FEM = ["affectionate", "cheer", "collab", "commit", "communal", "compassion", "connect", "considerate", "cooperat",
       "depend", "emotiona", "empath", "feminine", "flatterable", "gentle", "honest", "interpersonal", "interdependen",
       "kind", "kinship", "loyal", "modesty", "nag", "nurtur", "pleasant", "polite", "quiet", "respon", "sensitiv",
       "submissive", "support", "sympath", "tender", "together", "trust", "understand", "warm", "whin", "yield"]
_tok = re.compile(r"[a-z][a-z'-]+")


def _count(words, stems):
    return sum(1 for w in words if any(w.startswith(s) for s in stems))


def wording(df: pd.DataFrame, strip_title: bool = True) -> pd.DataFrame:
    """Masculine/feminine-coded word counts per description.

    strip_title=True drops the posting's own title words first: stems like "analy" and "support" otherwise fire on
    "Data Analyst" and "Customer Support Specialist" themselves, which measures the job name, not the wording.
    """
    out = []
    for d, t in zip(df.description, df.title):
        w = _tok.findall(d.lower())
        if strip_title:
            tw = set(_tok.findall(t.lower()))
            w = [x for x in w if x not in tw and not any(x.startswith(y[:6]) for y in tw if len(y) > 6)]
        out.append((_count(w, MASC), _count(w, FEM), len(w)))
    a = np.array(out)
    r = df[["job_id", "title"]].copy()
    r["masc"], r["fem"], r["words"] = a[:, 0], a[:, 1], a[:, 2]
    r["gskew"] = (r.masc - r.fem) / np.sqrt(r.words.clip(lower=1))
    r["family"] = r.title.str.replace(r"^(Senior|Junior|Lead) ", "", regex=True)
    return r


def pay(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["has_pay"] = d.min_salary.notna() & d.max_salary.notna()
    d["law"] = d.state.isin(LAW_STATES)
    d["width"] = d.max_salary / d.min_salary
    by_state = d.groupby("state").agg(postings=("job_id", "count"), disclosed=("has_pay", "mean")).reset_index()
    by_state["law"] = by_state.state.isin(LAW_STATES)
    wide = d[d.has_pay & (d.width > 2)]
    return {
        "disclosed_overall": float(d.has_pay.mean()),
        "disclosed_law_states": float(d[d.law].has_pay.mean()),
        "disclosed_other_states": float(d[~d.law].has_pay.mean()),
        "law_state_gap_postings": int((d.law & ~d.has_pay).sum()),
        "wide_range_share": float(len(wide) / max(1, d.has_pay.sum())),
        "by_state": by_state.sort_values("disclosed").round(4).to_dict("records"),
        "by_family": (d.assign(family=d.title.str.replace(r"^(Senior|Junior|Lead) ", "", regex=True)).groupby("family")
                      .has_pay.mean().sort_values().round(4).to_dict()),
    }


def reposts(df: pd.DataFrame, dup_clusters_same_company: list[list[int]] | None = None) -> dict:
    """Repost behaviour per (company, title, location) role key. Two signals:
    1. explicit: original_listed_time earlier than listed_time (the platform links the repost)
    2. hidden: same role key listed again under a new job_id after the earlier listing expired (found via
       near-duplicate text within the same company, or an identical key)
    This measures how often a role is relisted, not whether it is real: no ground truth for "ghost jobs" exists."""
    d = df.copy()
    d["explicit"] = d.original_listed_time < d.listed_time
    d["key"] = d.company_name + "|" + d.title + "|" + d.location
    g = d.sort_values("listed_time").groupby("key")
    per_role = g.agg(company=("company_name", "first"), title=("title", "first"), listings=("job_id", "count"),
                     first=("listed_time", "min"), last=("listed_time", "max"), explicit=("explicit", "sum")).reset_index()
    per_role["span_days"] = (per_role["last"] - per_role["first"]).dt.days
    per_role["relisted"] = per_role.listings > 1
    perennial = per_role[(per_role.listings >= 4) & (per_role.span_days >= 90)]
    comp = per_role.groupby("company").agg(roles=("key", "count"), relisted_share=("relisted", "mean"),
                                           listings=("listings", "sum")).reset_index()
    comp = comp[comp.roles >= 10].sort_values("relisted_share", ascending=False)
    gaps = d.sort_values("listed_time").groupby("key").listed_time.diff().dropna().dt.days
    return {
        "roles": int(len(per_role)), "relisted_share": float(per_role.relisted.mean()),
        "explicit_repost_share_of_listings": float(d.explicit.mean()),
        "hidden_relist_listings": int(((d.sort_values("listed_time").groupby("key").cumcount() > 0) & ~d.sort_values("listed_time").explicit).sum()),
        "median_days_between_relists": float(gaps.median()) if len(gaps) else None,
        "perennial_roles": int(len(perennial)),
        "listings_hist": per_role.listings.value_counts().sort_index().to_dict(),
        "top_companies": comp.head(10).round(3).to_dict("records"),
        "perennial_examples": perennial.sort_values("listings", ascending=False).head(10)[["company", "title", "listings", "span_days"]].to_dict("records"),
    }
