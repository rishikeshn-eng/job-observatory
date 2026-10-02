"""Synthetic postings in the schema of Kaggle's "LinkedIn Job Postings (2023-2024)" (job_id, company_name, title,
description, location, min/max_salary, pay_period, listed_time, original_listed_time, expiry, ...).

NOT real data. Planted ground truth: staffing-agency postings copied across client companies, repost chains (some
with a new job_id and no link to the original = hidden reposts), and a long shared boilerplate paragraph in most
postings (the classic dedupe confounder). Pay disclosure is more likely in pay-transparency states but not universal.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

FAMILIES = {
    "Software Engineer": dict(masc=0.45, fem=0.12, pay=0.7),
    "Data Analyst": dict(masc=0.25, fem=0.18, pay=0.6),
    "Registered Nurse": dict(masc=0.05, fem=0.5, pay=0.75),
    "Sales Representative": dict(masc=0.6, fem=0.08, pay=0.35),
    "Warehouse Associate": dict(masc=0.35, fem=0.05, pay=0.8),
    "Customer Support Specialist": dict(masc=0.08, fem=0.45, pay=0.5),
    "Marketing Manager": dict(masc=0.3, fem=0.25, pay=0.45),
    "Accountant": dict(masc=0.15, fem=0.2, pay=0.55),
    "Electrician": dict(masc=0.4, fem=0.05, pay=0.7),
    "Teacher": dict(masc=0.05, fem=0.45, pay=0.65),
}
MASC = ["competitive", "dominant", "aggressive", "driven", "ambitious", "independent", "decisive", "assertive", "fearless", "self-reliant"]
FEM = ["supportive", "collaborative", "nurturing", "empathetic", "compassionate", "cooperative", "committed", "warm", "understanding", "interpersonal"]
STATES = {"CA": 0.9, "NY": 0.88, "CO": 0.92, "WA": 0.9, "TX": 0.4, "FL": 0.38, "IL": 0.55, "OH": 0.35, "GA": 0.34, "NC": 0.36}
LAW = {"CA", "NY", "CO", "WA"}
SENT = [
    "You will work closely with cross-functional teams to deliver results for our customers.",
    "We offer a flexible schedule, health coverage and a learning budget.",
    "Responsibilities include planning, execution and reporting against quarterly goals.",
    "Candidates should have strong written and verbal communication skills.",
    "The role reports to the department head and sits in our regional office.",
    "You will own projects from kickoff through delivery and handover.",
    "Experience with modern tooling and a track record of shipping is a plus.",
    "Our team values curiosity, ownership and clear documentation.",
    "Day to day you will review requests, prioritise the queue and follow up with stakeholders.",
    "We are growing quickly and building processes as we go.",
]
BOILER = ("We are an equal opportunity employer and value diversity at our company. We do not discriminate on the basis "
          "of race, religion, colour, national origin, gender, sexual orientation, age, marital status, veteran status, "
          "or disability status. All qualified applicants will receive consideration for employment.")
COMPANY_WORDS = ["Apex", "Bright", "Cedar", "Delta", "Ember", "Forge", "Granite", "Harbor", "Ion", "Juniper", "Keystone", "Lumen",
                 "Meridian", "Nova", "Orbit", "Pine", "Quartz", "Ridge", "Summit", "Tidal"]


def _desc(rng, fam, company):
    p = FAMILIES[fam]
    s = rng.sample(SENT, 5)
    if rng.random() < p["masc"]:
        s.insert(rng.randint(0, 5), f"We want a {rng.choice(MASC)} {rng.choice(MASC)} person for this {fam.lower()} role.")
    if rng.random() < p["fem"]:
        s.insert(rng.randint(0, 5), f"We want a {rng.choice(FEM)} {rng.choice(FEM)} person for this {fam.lower()} role.")
    body = f"{company} is hiring a {fam}. " + " ".join(s)
    return body + (" " + BOILER if rng.random() < 0.7 else "")


def generate(n_roles: int = 9000, seed: int = 21):
    rng = random.Random(seed)
    now = datetime(2024, 4, 1)
    companies = [f"{rng.choice(COMPANY_WORDS)} {rng.choice(['Labs', 'Health', 'Systems', 'Group', 'Works', 'Logistics'])} {i}" for i in range(300)]
    agencies = companies[:12]  # staffing agencies
    repost_p = {c: (rng.betavariate(0.6, 3) if c not in agencies else rng.betavariate(2, 3)) for c in companies}
    rows, truth_repost = [], []
    jid = 1000
    for role in range(n_roles):
        company = rng.choice(companies)
        fam = rng.choice(list(FAMILIES))
        title = f"{rng.choice(['', 'Senior ', 'Junior ', 'Lead '])}{fam}".strip()
        state = rng.choices(list(STATES), weights=[1] * len(STATES))[0]
        city = f"{state}-city{rng.randint(1, 4)}"
        base = _desc(rng, fam, company)
        disclose_p = FAMILIES[fam]["pay"] * (STATES[state] if state in LAW else 0.55 * STATES[state] + 0.2)
        has_pay = rng.random() < min(0.97, disclose_p)
        lo = rng.randint(30, 110) * 1000
        wide = rng.choice([1.2, 1.3, 1.5, 2.2, 3.0])
        t0 = now - timedelta(days=rng.randint(0, 150))
        n_listings = 1
        while rng.random() < repost_p[company] and n_listings < 8:
            n_listings += 1
        if company in agencies and rng.random() < 0.02:
            n_listings = 6  # perennial listing
        chain = []
        for k in range(n_listings):
            listed = t0 + timedelta(days=31 * k + rng.randint(0, 3))
            if listed > now:
                break
            hidden = k > 0 and rng.random() < 0.35  # new job_id AND no original_listed_time link
            desc = base if rng.random() < 0.8 else base.replace("our customers", "our clients")
            jid += 1
            rows.append({"job_id": jid, "company_name": company, "title": title, "description": desc,
                         "location": city, "state": state, "work_type": rng.choice(["Full-time", "Full-time", "Contract"]),
                         "remote_allowed": rng.random() < 0.25,
                         "min_salary": lo if has_pay else None, "max_salary": round(lo * wide) if has_pay else None,
                         "pay_period": "YEARLY" if has_pay else None,
                         "listed_time": listed, "original_listed_time": (t0 if (k > 0 and not hidden) else listed),
                         "expiry": listed + timedelta(days=30), "role_id": role})
            chain.append(jid)
        if not chain:
            continue
        if len(chain) > 1:
            truth_repost.append(chain)
        # agency copies the role under client names
        if company in agencies and rng.random() < 0.55:
            clients = rng.sample(companies[12:], rng.randint(1, 4))
            for cl in clients:
                jid += 1
                d = rows[-1]["description"].replace(company, cl)
                rows.append({**rows[-1], "job_id": jid, "company_name": cl, "description": d, "role_id": role})
    df = pd.DataFrame(rows)
    # cross-company duplicate truth: rows of one role posted under different company names (agency + client copies)
    dup = []
    for _, g in df.groupby("role_id"):
        if g.company_name.nunique() > 1:
            ids = g.job_id.tolist()
            cos = g.company_name.tolist()
            dup += [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)) if cos[i] != cos[j]]
    return df, {"dup_pairs": dup, "repost_chains": truth_repost}
