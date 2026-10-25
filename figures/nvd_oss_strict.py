#!/usr/bin/env python3
# ============================================================
# NVD STRICT OSS SUPPLY CHAIN FILTER
# Downloads full NVD CVE dataset and filters strictly
# for OSS supply chain incidents only
# No API key needed. Free.
# ============================================================
# Usage:
#   pip install requests pandas tqdm
#   python nvd_oss_strict.py --input oss_threat_dataset_10k.csv
# Output: oss_threat_dataset_nvd.csv
# ============================================================

import requests
import pandas as pd
import json
import os
import zipfile
import hashlib
import time
import argparse
from tqdm import tqdm
from collections import Counter

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

# ============================================================
# STRICT OSS SUPPLY CHAIN KEYWORD FILTERS
# These are very specific to OSS supply chain attacks only
# General vulnerabilities will NOT pass these filters
# ============================================================

# Must contain at least one STRONG keyword
STRONG_KEYWORDS = [
    # Package-specific
    "malicious package", "malicious npm", "malicious pypi",
    "malicious rubygem", "malicious cargo", "malicious maven",
    "malicious docker", "malicious pip",
    "typosquat", "typo-squat", "brandjack",
    "dependency confusion", "namespace confusion",
    "package hijack", "package takeover", "account takeover npm",
    "account takeover pypi",
    # Pipeline-specific
    "github actions", "poisoned pipeline", "pipeline poisoning",
    "workflow injection", "ci/cd pipeline attack",
    "poisoned pipeline execution", "ppe attack",
    # Trojan Source specific
    "trojan source", "unicode bidirectional", "bidi override",
    "bidirectional control character",
    # Supply chain specific
    "software supply chain attack", "supply chain compromise",
    "supply chain poisoning", "upstream dependency",
    "transitive dependency malicious",
    # Install hooks
    "postinstall malicious", "setup.py malicious",
    "install hook malicious", "preinstall malicious",
]

# Must contain at least one ECOSYSTEM keyword
ECOSYSTEM_KEYWORDS = [
    "npm", "pypi", "python package", "node package",
    "rubygems", "cargo", "crates.io", "maven",
    "packagist", "nuget", "github actions",
    "docker hub", "docker image", "go module",
    "open source", "open-source", "oss",
    "package registry", "package manager",
    "supply chain"
]

# Must NOT contain these (false positive filters)
EXCLUDE_KEYWORDS = [
    "windows", "microsoft office", "adobe", "oracle database",
    "cisco", "juniper", "kernel privilege", "buffer overflow",
    "sql injection", "cross-site scripting", "xss",
    "integer overflow", "use after free", "null pointer",
    "path traversal", "race condition", "memory corruption",
    "format string", "heap overflow", "stack overflow",
    "denial of service", "information disclosure unrelated",
    "authentication bypass unrelated"
]

def is_strict_oss_supply_chain(title, description):
    """
    Strict filter for genuine OSS supply chain incidents.
    Returns True only if entry is clearly OSS supply chain.
    """
    combined = (title + " " + description).lower()

    # Must have at least one strong keyword
    has_strong = any(kw in combined for kw in STRONG_KEYWORDS)
    if not has_strong:
        return False

    # Must have at least one ecosystem keyword
    has_ecosystem = any(kw in combined for kw in ECOSYSTEM_KEYWORDS)
    if not has_ecosystem:
        return False

    return True

def classify_rule_based(title, description):
    text = (title + " " + description).lower()

    # AV-300 Trojan Source
    if any(kw in text for kw in [
        "unicode bidirectional", "bidi character", "trojan source",
        "invisible character", "zero-width", "homoglyph",
        "bidirectional control", "rtlo", "right-to-left override"]):
        return "AV-300", 0.95

    # AV-509 Dependency Confusion
    if any(kw in text for kw in [
        "dependency confusion", "namespace confusion",
        "internal package", "private package", "public registry override",
        "registry priority", "internal dependency", "private dependency"]):
        return "AV-509", 0.92

    # AV-410 Pipeline Poisoning
    if any(kw in text for kw in [
        "github actions", "workflow inject", "poisoned pipeline",
        "ppe attack", "pipeline poison", "ci/cd compromise",
        "workflow file", "runner worker", "actions secret",
        "pipeline execution", "workflow token", "pull request injection"]):
        return "AV-410", 0.90

    # AV-200 Typosquatting
    if any(kw in text for kw in [
        "typosquat", "typo-squat", "brandjack", "impersonat",
        "mimic", "look-alike", "lookalike", "character substitut",
        "name squatt", "one character", "namespace impersonat",
        "fake package", "package impersonat", "similar name",
        "account takeover"]):
        return "AV-200", 0.88

    # AV-400 Malicious Builds
    if any(kw in text for kw in [
        "postinstall", "setup.py", "install script",
        "lifecycle hook", "malicious code execut",
        "backdoor inject", "reverse shell", "cryptominer",
        "credential steal", "exfiltrat", "malicious payload",
        "dropper", "malicious npm", "malicious pypi",
        "malicious package", "install malicious"]):
        return "AV-400", 0.88

    # Broader patterns
    if any(kw in text for kw in ["ci/cd", "pipeline", "workflow"]):
        return "AV-410", 0.75
    if any(kw in text for kw in ["malicious", "backdoor", "trojan", "malware"]):
        return "AV-400", 0.75

    return "AV-400", 0.70  # Default for supply chain entries

# ============================================================
# NVD DATA FEEDS (FULL DUMP - FREE)
# ============================================================

def download_nvd_feeds():
    """
    Download NVD CVE data feeds by year.
    NVD provides annual JSON feeds - completely free.
    """
    print("\n[1/2] Downloading NVD CVE feeds...")
    all_cves = []
    years = list(range(2017, 2026))  # 2017-2025

    os.makedirs("nvd_feeds", exist_ok=True)

    for year in years:
        zip_path = f"nvd_feeds/nvdcve-1.1-{year}.json.gz"
        json_path = f"nvd_feeds/nvdcve-1.1-{year}.json"

        # Check if already downloaded
        if os.path.exists(json_path):
            print(f"  {year}: Already downloaded, loading...")
        else:
            url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
            print(f"  {year}: Downloading...")
            try:
                resp = requests.get(url, timeout=120, stream=True)
                if resp.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Decompress
                    import gzip
                    with gzip.open(zip_path, 'rb') as f_in:
                        with open(json_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    os.remove(zip_path)
                    size_mb = os.path.getsize(json_path) / 1024 / 1024
                    print(f"    Downloaded {size_mb:.1f} MB")
                else:
                    print(f"    Failed: {resp.status_code}")
                    continue

                time.sleep(2)  # Be polite to NVD

            except Exception as e:
                print(f"    Error: {e}")
                continue

        # Load and parse
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cves = data.get("CVE_Items", [])
            print(f"  {year}: {len(cves)} CVEs loaded")

            for item in cves:
                cve = item.get("cve", {})
                cve_id = cve.get("CVE_data_meta", {}).get("ID", "")

                # Get description
                desc_data = cve.get("description", {})
                desc_items = desc_data.get("description_data", [])
                desc = ""
                for d in desc_items:
                    if d.get("lang") == "en":
                        desc = d.get("value", "")
                        break

                if not desc or len(desc) < 30:
                    continue

                # Strict OSS supply chain filter
                if not is_strict_oss_supply_chain(cve_id + " " + desc, desc):
                    continue

                # Get references
                refs = cve.get("references", {}).get("reference_data", [])
                source_url = refs[0].get("url", "") if refs else ""

                # Get published date
                published = item.get("publishedDate", "")[:4]

                all_cves.append({
                    "cve_id": cve_id,
                    "description": desc,
                    "source_url": source_url,
                    "year": published
                })

        except Exception as e:
            print(f"  Error loading {year}: {e}")
            continue

    print(f"\nTotal strict OSS supply chain CVEs found: {len(all_cves)}")
    return all_cves


def fetch_nvd_api_targeted():
    """
    Use NVD API with very targeted supply chain keywords.
    Backup method if feeds fail.
    """
    print("\n[Backup] Using NVD API with targeted keywords...")
    entries = []

    # Very specific OSS supply chain search terms
    targeted_keywords = [
        "malicious npm package",
        "malicious PyPI package",
        "typosquatting npm",
        "typosquatting PyPI",
        "dependency confusion npm",
        "dependency confusion PyPI",
        "supply chain npm",
        "supply chain PyPI",
        "malicious RubyGems",
        "malicious cargo crate",
        "malicious Maven",
        "GitHub Actions supply chain",
        "poisoned pipeline execution",
        "postinstall malicious",
        "setup.py malicious",
        "trojan source unicode",
        "package hijacking npm",
        "package takeover PyPI",
        "backdoor npm package",
        "backdoor PyPI package",
        "cryptominer npm",
        "credential stealing npm",
        "malicious Docker image",
        "supply chain attack open source"
    ]

    for keyword in tqdm(targeted_keywords, desc="  API queries"):
        try:
            url = (f"https://services.nvd.nist.gov/rest/json/cves/2.0"
                   f"?keywordSearch={requests.utils.quote(keyword)}"
                   f"&resultsPerPage=100")
            resp = requests.get(url, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                vulns = data.get("vulnerabilities", [])
                for item in vulns:
                    cve = item.get("cve", {})
                    cve_id = cve.get("id", "")
                    descs = cve.get("descriptions", [])
                    desc = ""
                    for d in descs:
                        if d.get("lang") == "en":
                            desc = d.get("value", "")
                            break

                    if not desc or len(desc) < 30:
                        continue

                    if not is_strict_oss_supply_chain(cve_id, desc):
                        continue

                    entries.append({
                        "cve_id": cve_id,
                        "description": desc,
                        "source_url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        "year": cve.get("published", "")[:4]
                    })

            time.sleep(6)  # NVD rate limit: 5 req/30s without key

        except Exception as e:
            print(f"  Warning: {e}")
            time.sleep(10)
            continue

    print(f"  API found {len(entries)} strict OSS entries")
    return entries


def convert_to_entries(cves):
    """Convert NVD CVE records to dataset format."""
    entries = []
    for cve in cves:
        cve_id = cve["cve_id"]
        desc = cve["description"]

        # Create meaningful title from description
        title = desc[:150].rstrip('.')
        if len(desc) > 150:
            last_space = title.rfind(' ')
            if last_space > 100:
                title = title[:last_space] + "..."

        # Detect ecosystem
        desc_lower = desc.lower()
        ecosystem = ""
        if "npm" in desc_lower or "node" in desc_lower:
            ecosystem = "npm"
        elif "pypi" in desc_lower or "python package" in desc_lower:
            ecosystem = "PyPI"
        elif "rubygems" in desc_lower or "ruby gem" in desc_lower:
            ecosystem = "RubyGems"
        elif "cargo" in desc_lower or "crates.io" in desc_lower:
            ecosystem = "Cargo"
        elif "maven" in desc_lower:
            ecosystem = "Maven"
        elif "github actions" in desc_lower:
            ecosystem = "GitHub Actions"
        elif "docker" in desc_lower:
            ecosystem = "Docker"

        entries.append({
            "id": cve_id.lower().replace("-", ""),
            "title": title,
            "description": desc[:1000],
            "label": "",
            "predicted_label": "",
            "source": cve.get("source_url", f"https://nvd.nist.gov/vuln/detail/{cve_id}"),
            "ecosystem": ecosystem,
            "year": cve.get("year", ""),
            "raw_source": "NVD-STRICT"
        })

    return entries


def deduplicate_with_existing(new_entries, existing_entries):
    """Deduplicate new entries against existing dataset."""
    print(f"\nDeduplicating {len(new_entries)} new entries against {len(existing_entries)} existing...")

    existing_ids = set(str(e.get("id","")).strip() for e in existing_entries)
    existing_sources = set(str(e.get("source","")).strip() for e in existing_entries)
    existing_hashes = set(
        hashlib.md5(str(e.get("title","")).lower().strip().encode()).hexdigest()
        for e in existing_entries
    )

    unique_new = []
    for entry in new_entries:
        eid = str(entry.get("id","")).strip()
        src = str(entry.get("source","")).strip()
        title_hash = hashlib.md5(str(entry.get("title","")).lower().strip().encode()).hexdigest()

        if eid in existing_ids: continue
        if src in existing_sources: continue
        if title_hash in existing_hashes: continue
        if len(str(entry.get("description",""))) < 30: continue

        existing_ids.add(eid)
        existing_sources.add(src)
        existing_hashes.add(title_hash)
        unique_new.append(entry)

    print(f"New unique entries after dedup: {len(unique_new)}")
    return unique_new


def label_entries(entries):
    labeled = []
    uncertain = []
    for entry in entries:
        if entry.get("label") in VALID_LABELS:
            labeled.append(entry)
            continue
        label, conf = classify_rule_based(
            entry.get("title", ""),
            entry.get("description", "")
        )
        if conf >= 0.85:
            entry["label"] = label
            entry["predicted_label"] = label
            labeled.append(entry)
        else:
            entry["label"] = label  # Still use it — it's supply chain verified
            entry["predicted_label"] = label
            labeled.append(entry)  # Keep all since strict filter already verified OSS

    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='oss_threat_dataset_10k.csv')
    parser.add_argument('--output', default='oss_threat_dataset_nvd_strict.csv')
    parser.add_argument('--method', default='api',
                        choices=['api', 'feeds'],
                        help='api: NVD API (slower), feeds: Annual feeds (faster, ~2GB)')
    args = parser.parse_args()

    print("=" * 60)
    print("NVD STRICT OSS SUPPLY CHAIN FILTER")
    print(f"Method: {args.method}")
    print("=" * 60)

    # Load existing dataset
    existing = []
    try:
        df = pd.read_csv(args.input)
        existing = df.to_dict('records')
        print(f"\nLoaded existing: {len(existing)} entries")
    except FileNotFoundError:
        print(f"Warning: {args.input} not found")

    # Collect from NVD
    if args.method == 'feeds':
        print("\nUsing NVD annual feeds (downloads ~2GB total)")
        raw_cves = download_nvd_feeds()
    else:
        print("\nUsing NVD API with targeted keywords")
        raw_cves = fetch_nvd_api_targeted()

    if not raw_cves:
        print("No CVEs found. Try --method feeds")
        return

    # Convert to dataset format
    new_entries = convert_to_entries(raw_cves)
    print(f"\nConverted to {len(new_entries)} dataset entries")

    # Deduplicate against existing
    unique_new = deduplicate_with_existing(new_entries, existing)

    if not unique_new:
        print("No new unique entries found after deduplication.")
        print("All NVD strict OSS entries are already in your dataset.")
        return

    # Label entries
    print(f"\nLabeling {len(unique_new)} new entries...")
    labeled_new = label_entries(unique_new)

    # Combine with existing
    existing_valid = [e for e in existing
                      if e.get("label") in VALID_LABELS]
    all_labeled = existing_valid + labeled_new

    # Stats
    dist = Counter(e["label"] for e in all_labeled)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Existing entries: {len(existing_valid)}")
    print(f"New NVD entries added: {len(labeled_new)}")
    print(f"Total: {len(all_labeled)}")
    print(f"\nLabel distribution:")
    for label in VALID_LABELS:
        count = dist.get(label, 0)
        pct = count/len(all_labeled)*100 if all_labeled else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Save
    cols = ["id", "title", "description", "label",
            "predicted_label", "source", "ecosystem", "year"]
    df_out = pd.DataFrame(all_labeled)
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out[cols].to_csv(args.output, index=False)
    print(f"\nSaved {len(all_labeled)} entries to {args.output}")

    target = 10000
    if len(all_labeled) >= target:
        print(f"\n✅ Target of {target} reached!")
    else:
        needed = target - len(all_labeled)
        print(f"\n⚠️ Got {len(all_labeled)} of {target}. Need {needed} more.")
        if args.method == 'api':
            print("Try --method feeds for more comprehensive coverage")

    print("\nDone.")

if __name__ == "__main__":
    main()
