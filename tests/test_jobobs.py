import pandas as pd

from jobobs import audits, dedupe, synth

A = "We are hiring a data engineer to build pipelines and own the warehouse. You will work with analysts daily and ship weekly."
BOILER = "We are an equal opportunity employer and value diversity at our company. All qualified applicants will receive consideration."


def test_near_duplicates_found_unrelated_not():
    texts = [A + " " + BOILER, A.replace("data engineer", "data engineer II") + " " + BOILER,
             "Looking for a registered nurse to join our night shift team at the regional hospital. " + BOILER]
    pairs, _ = dedupe.find_duplicates(texts, threshold=0.5, max_df=None)
    assert (0, 1) in pairs and not any(2 in p for p in pairs)


def test_boilerplate_filter_removes_false_pairs():
    # many unrelated postings that share one long paragraph
    long_boiler = " ".join(f"clause{k} of the standard legal paragraph" for k in range(12))  # ~72 words, identical everywhere
    texts = [f"Role {i} unique topic{i} alpha{i} beta{i} gamma{i}. " + long_boiler for i in range(60)]
    raw, _ = dedupe.find_duplicates(texts, threshold=0.5, max_df=None)
    filt, _ = dedupe.find_duplicates(texts, threshold=0.5, max_df=0.2)
    assert len(raw) > 100 and len(filt) == 0


def test_clusters_union_find():
    assert sorted(map(sorted, dedupe.clusters(6, [(0, 1), (1, 2), (4, 5)]))) == [[0, 1, 2], [4, 5]]


def test_synth_ground_truth_and_dedupe_quality():
    df, truth = synth.generate(1200, seed=3)
    assert truth["dup_pairs"] and truth["repost_chains"]
    idx = {j: i for i, j in enumerate(df.job_id)}
    pairs, _ = dedupe.find_duplicates(df.description.tolist(), 0.6, 0.05)
    found = {tuple(sorted(p)) for p in pairs if df.company_name[p[0]] != df.company_name[p[1]]}
    t = {tuple(sorted((idx[a], idx[b]))) for a, b in truth["dup_pairs"]}
    assert len(found & t) / len(found) > 0.85 and len(found & t) / len(t) > 0.85


def test_pay_audit_counts_law_states():
    df = pd.DataFrame({"job_id": range(4), "state": ["CA", "CA", "TX", "TX"], "title": ["A"] * 4,
                       "min_salary": [1.0, None, 1.0, None], "max_salary": [3.0, None, 1.5, None]})
    p = audits.pay(df)
    assert p["disclosed_law_states"] == 0.5 and p["law_state_gap_postings"] == 1 and p["wide_range_share"] == 0.5


def test_title_words_do_not_count_as_coded_words():
    df = pd.DataFrame({"job_id": [1], "title": ["Customer Support Specialist"],
                       "description": ["We are hiring a Customer Support Specialist. Support our customers daily."]})
    assert audits.wording(df, strip_title=False).fem[0] >= 2
    assert audits.wording(df, strip_title=True).fem[0] == 0


def test_repost_explicit_and_perennial():
    base = pd.Timestamp("2024-01-01")
    rows = [{"job_id": i, "company_name": "Acme", "title": "Dev", "location": "X", "listed_time": base + pd.Timedelta(days=31 * i),
             "original_listed_time": base} for i in range(5)]
    r = audits.reposts(pd.DataFrame(rows))
    assert r["roles"] == 1 and r["relisted_share"] == 1.0 and r["perennial_roles"] == 1
    assert r["explicit_repost_share_of_listings"] == 0.8
