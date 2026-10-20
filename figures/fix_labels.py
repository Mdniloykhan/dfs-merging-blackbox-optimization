#!/usr/bin/env python3
# ============================================================
# QUICK FIX — Relabel with lower confidence threshold
# Also fixes OSV.dev API and relaxes deduplication
# Run: python fix_labels.py
# ============================================================

import requests
import pandas as pd
import hashlib
import time
import os
from collections import Counter

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

def classify_rule_based(title, description):
    text = (title + " " + description).lower()

    # AV-300 — Unicode/Trojan Source
    av300 = ["unicode bidirectional","bidi character","trojan source",
             "invisible character","zero-width","homoglyph",
             "bidirectional control","rtlo","right-to-left override",
             "cve-2021-42574","unicode obfuscat"]
    if any(kw in text for kw in av300):
        return "AV-300", 0.95

    # AV-509 — Dependency Confusion
    av509 = ["dependency confusion","namespace confusion",
             "internal package","private package","public registry override",
             "package resolution","registry priority",
             "internal dependency","private dependency","confused deputy"]
    if any(kw in text for kw in av509):
        return "AV-509", 0.90

    # AV-410 — Pipeline Poisoning
    av410 = ["github actions","workflow inject","poisoned pipeline",
             "ppe attack","pipeline poison","ci/cd compromise",
             "workflow file","runner worker","github action compromise",
             "actions secret","pipeline execution","workflow token",
             "pull request injection","pr injection","pipeline execution",
             "cicd secret","build pipeline compromis"]
    if any(kw in text for kw in av410):
        return "AV-410", 0.90

    # AV-200 — Typosquatting
    av200 = ["typosquat","typo-squat","misspell","impersonat",
             "mimic","look-alike","lookalike","similar name",
             "character substitut","brandjack","name squatt",
             "one character","one-character","namespace impersonat",
             "fake package","malicious package name","copied package",
             "package impersonat"]
    if any(kw in text for kw in av200):
        return "AV-200", 0.88

    # AV-400 — Malicious Builds
    av400_strong = ["postinstall","setup.py","install script",
                    "lifecycle hook","malicious code execut",
                    "code execut on install","backdoor inject",
                    "reverse shell","cryptominer","credential steal",
                    "exfiltrat","malicious payload","dropper",
                    "npm install malicious","pip install malicious"]
    if any(kw in text for kw in av400_strong):
        return "AV-400", 0.88

    # Broader patterns — lower confidence
    if any(kw in text for kw in ["ci/cd","pipeline","workflow","continuous integration"]):
        return "AV-410", 0.72
    if any(kw in text for kw in ["malicious","backdoor","trojan","malware","obfuscat"]):
        return "AV-400", 0.70
    if any(kw in text for kw in ["package","npm","pypi","registry","dependency"]):
        return "AV-400", 0.65

    return "UNCERTAIN", 0.0

def fetch_osv_fixed():
    """Fixed OSV.dev using correct API format."""
    print("\nFetching OSV.dev with fixed API...")
    entries = []
    ecosystems = ["npm", "PyPI", "RubyGems", "crates.io", "Maven"]

    for eco in ecosystems:
        print(f"  Trying {eco}...")
        try:
            # Use the correct OSV API v1 format
            resp = requests.post(
                "https://api.osv.dev/v1/query",
                json={"package": {"ecosystem": eco}, "page_size": 100},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            print(f"    Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                vulns = data.get("vulns", [])
                print(f"    Got {len(vulns)} vulns")

                supply_kws = ["malicious","backdoor","typosquat",
                              "supply chain","postinstall","setup.py",
                              "exfiltrat","cryptominer","reverse shell",
                              "dependency confusion","unicode","bidirectional"]

                for vuln in vulns:
                    vid = vuln.get("id","")
                    summary = vuln.get("summary","")
                    details = vuln.get("details","")
                    combined = (summary + " " + (details or "")).lower()

                    if any(kw in combined for kw in supply_kws):
                        affected = vuln.get("affected",[])
                        pkg_name = ""
                        if affected:
                            pkg_name = affected[0].get("package",{}).get("name","")

                        title = f"{pkg_name}: {summary[:150]}" if pkg_name else summary[:200]
                        entries.append({
                            "id": f"osv{vid.lower().replace('-','').replace('.','')}",
                            "title": title[:200],
                            "description": (details or summary)[:1000],
                            "label": "",
                            "predicted_label": "",
                            "source": f"https://osv.dev/vulnerability/{vid}",
                            "ecosystem": eco,
                            "year": vuln.get("published","")[:4],
                            "raw_source": "OSV"
                        })
            else:
                print(f"    Error: {resp.text[:200]}")
            time.sleep(2)
        except Exception as e:
            print(f"    Exception: {e}")

    print(f"  OSV total: {len(entries)}")
    return entries

def main():
    print("=" * 60)
    print("LABEL FIX + OSV RETRY")
    print("=" * 60)

    # Load all existing files
    all_entries = []

    # Load original 999
    try:
        df1 = pd.read_csv('oss_threat_dataset_beta.csv')
        all_entries.extend(df1.to_dict('records'))
        print(f"Loaded original: {len(df1)} entries")
    except: pass

    # Load high confidence from v3
    try:
        df2 = pd.read_csv('oss_threat_dataset_5k.csv')
        all_entries.extend(df2.to_dict('records'))
        print(f"Loaded v3 high conf: {len(df2)} entries")
    except: pass

    # Load needs review from v3
    try:
        df3 = pd.read_csv('oss_threat_dataset_5k_needs_review.csv')
        all_entries.extend(df3.to_dict('records'))
        print(f"Loaded v3 review: {len(df3)} entries")
    except: pass

    print(f"Total before dedup: {len(all_entries)}")

    # Deduplicate by source URL only (more relaxed)
    seen_sources = set()
    seen_ids = set()
    unique = []
    for e in all_entries:
        src = str(e.get("source","")).strip()
        eid = str(e.get("id","")).strip()
        if src and src in seen_sources:
            continue
        if eid and eid in seen_ids:
            continue
        if len(str(e.get("description",""))) < 20:
            continue
        if src: seen_sources.add(src)
        if eid: seen_ids.add(eid)
        unique.append(e)

    print(f"After relaxed dedup: {len(unique)}")

    # Fetch OSV with fixed API
    osv_entries = fetch_osv_fixed()
    unique.extend(osv_entries)
    print(f"After adding OSV: {len(unique)}")

    # Relabel everything with lower threshold
    print("\nRelabeling with lower confidence threshold (0.70)...")
    labeled = []
    review = []
    dist = Counter()

    for e in unique:
        current_label = str(e.get("label",""))

        # Keep already verified labels
        if current_label in VALID_LABELS:
            labeled.append(e)
            dist[current_label] += 1
            continue

        # Reclassify
        title = str(e.get("title",""))
        desc = str(e.get("description",""))
        label, conf = classify_rule_based(title, desc)

        if label != "UNCERTAIN" and conf >= 0.70:
            e["label"] = label
            e["predicted_label"] = label
            labeled.append(e)
            dist[label] += 1
        else:
            e["label"] = "REVIEW_NEEDED"
            review.append(e)

    print(f"\n--- FINAL RESULTS ---")
    print(f"Labeled entries: {len(labeled)}")
    print(f"Needs manual review: {len(review)}")
    print(f"\nLabel distribution:")
    for label in VALID_LABELS:
        count = dist.get(label,0)
        pct = count/len(labeled)*100 if labeled else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Save
    cols = ["id","title","description","label",
            "predicted_label","source","ecosystem","year"]

    df_out = pd.DataFrame(labeled)
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out[cols].to_csv("oss_threat_dataset_5k_fixed.csv", index=False)
    print(f"\nSaved {len(labeled)} entries to oss_threat_dataset_5k_fixed.csv")

    if review:
        df_rev = pd.DataFrame(review)
        for col in cols:
            if col not in df_rev.columns:
                df_rev[col] = ""
        df_rev[cols].to_csv("oss_threat_dataset_review.csv", index=False)
        print(f"Saved {len(review)} review entries to oss_threat_dataset_review.csv")

    target = 5000
    if len(labeled) >= target:
        print(f"\n✅ Target of {target} reached!")
    else:
        print(f"\n⚠️ Got {len(labeled)} of {target}.")
        print("Run expand_dataset_v3.py again for more entries.")

    print("Done.")

if __name__ == "__main__":
    main()
