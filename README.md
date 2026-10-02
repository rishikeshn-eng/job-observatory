# job-observatory

Four audits over job postings: **cross-company duplicates**, **repost behaviour**, **pay disclosure**, and **gendered wording**.

**Live UI:** https://rishikeshn-eng.github.io/job-observatory/

> **Status: published numbers come from synthetic postings, not the Kaggle LinkedIn data.**
> The Kaggle files need a login and a download, so `jobobs/synth.py` generates 10,933 postings from 300 companies in the schema of *LinkedIn Job Postings (2023-2024)* (`job_id`, `company_name`, `title`, `description`, `min_salary`, `max_salary`, `listed_time`, `original_listed_time`, `expiry`, ...) with planted ground truth: staffing-agency postings copied across client names, repost chains, a shared boilerplate paragraph, and pay and wording patterns. The methods are real and tested; the shares in the UI describe the generator, not the labour market. Run on the Kaggle CSV with `python -m jobobs.run --csv postings.csv` (needs columns above; `state` is parsed from `location` if absent).

## What each audit does

1. **Duplicates across companies** (`dedupe.py`). Word 5-gram shingles, MinHash (64 permutations), LSH banding, candidates verified by exact Jaccard >= 0.6, union-find clusters. Shingles in more than 5% of postings are dropped first.
2. **Repost behaviour** (`audits.reposts`). Groups listings by company + title + location. Counts explicit reposts (`original_listed_time < listed_time`), relists under a new job ID with no platform link, perennial roles (4+ listings over 90+ days), and per-company relist share.
3. **Pay disclosure** (`audits.pay`). Disclosure rate by state and role family, gap in pay-range-law states (CA, NY, CO, WA: an illustrative set, verify against current statutes), share of ranges wider than 2x.
4. **Gendered wording** (`audits.wording`). Counts masculine- and feminine-coded word stems (after Gaucher, Friesen & Kay 2011; a partial transcription, check the paper's appendix) and reports a length-normalised skew per role family.

## What the synthetic run shows about the methods

| check | result |
|---|---|
| Dedupe **with** boilerplate filter | precision **95.8%**, recall 97.1% (1,250 cross-company pairs found, 1,234 planted) |
| Dedupe **without** it | precision **2.0%**: 63,127 pairs reported, because 70% of postings share an equal-opportunity paragraph |
| Same-company chain pairs recovered by text dedupe | precision 99.8%, recall 99.9% (easy here: the generator keeps titles identical; real relists vary titles) |
| Wording lexicon on titles | The stems `analy` and `support` fire on "Data Analyst" and "Customer Support Specialist" themselves: counting them moved those two families' skew from -0.04 to +0.12 and from -0.12 to -0.28. Title words are now excluded; the UI shows both columns |

The boilerplate result is the one to take away: naive near-duplicate detection on job text mostly finds shared legal text. The last row is a measurement artifact that any lexicon audit of job ads can hit.

## What this does not claim

- **Reposting is not fakeness.** There is no ground truth for "ghost jobs" in either Kaggle dataset. The audit reports how often roles are relisted, which has many legitimate causes (continuous hiring, turnover, expired requisitions). Treat the "perennial role" list as a review queue.
- **Pay-range gaps are not violations.** The data lacks employer size, remote scope and effective dates, which the laws turn on.
- **A wording score is a wording signal.** It does not show that a posting discriminates or deters applicants; the lexicon comes from lab studies of how ads are perceived.

## Run

```bash
pip install numpy pandas pytest
python -m pytest -q
python -m jobobs.run              # synthetic, ~20 s; writes docs/results.json
python scripts/build_site.py      # docs/index.html
```
