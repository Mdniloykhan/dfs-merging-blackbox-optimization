#!/usr/bin/env python3
# ============================================================
# AUTOMATED DATASET EXPANSION SCRIPT
# Grows OSS threat dataset from 999 to 10,000+ entries
# Sources: GitHub Security Advisories API + CISA KEV API
#          + OSV.dev API + PyPI Advisory DB
# Cost: ~$5-15 GPT-4 API calls for 10,000 labels
# Time: 4-6 hours depending on API rate limits
# ============================================================
# Usage:
#   pip install requests pandas openai tqdm
#   python expand_dataset.py --api-key YOUR_OPENAI_KEY
# Output: oss_threat_dataset_expanded.csv
# ============================================================

import requests
import pandas as pd
import json
import time
import argparse
import hashlib
import os
from tqdm import tqdm
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

TAXONOMY_PROMPT = """You are an expert in open source software supply chain security. 
Classify the following security incident into exactly one of these categories:

AV-200 (Typosquatting): Malicious packages that mimic legitimate package names through character substitution, transposition, or namespace impersonation.

AV-300 (Trojan Source): Attacks exploiting Unicode bidirectional control characters or homoglyphs to create source code that executes differently from its visual representation.

AV-400 (Malicious Builds): Packages containing malicious code executed during installation, build, or import processes through lifecycle hooks such as postinstall and setup.py.

AV-410 (Pipeline Poisoning): Attacks that compromise CI/CD pipeline execution including GitHub Actions workflow injection, poisoned pipeline execution (PPE), and build system compromise.

AV-509 (Dependency Confusion): Attacks exploiting package manager resolution priority to cause build systems to install malicious public packages instead of intended private internal dependencies.

Respond with ONLY the category label (AV-200, AV-300, AV-400, AV-410, or AV-509). No explanation."""

# ============================================================
# API CLIENTS
# ============================================================

def fetch_github_advisories(per_page=100, max_pages=50):
    """Fetch malicious package advisories from GitHub Security Advisories API."""
    print("\n[1/4] Fetching GitHub Security Advisories...")
    entries = []
    
    # Search for supply chain related advisories
    search_terms = [
        "malicious package",
        "typosquatting",
        "dependency confusion", 
        "pipeline poisoning",
        "supply chain",
        "trojan source",
        "backdoor npm",
        "backdoor pypi",
        "malicious npm",
        "malicious pypi"
    ]
    
    headers = {"Accept": "application/vnd.github+json"}
    # Add GitHub token if available
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    
    for term in search_terms:
        url = f"https://api.github.com/advisories?type=malware&per_page={per_page}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                advisories = resp.json()
                for adv in advisories:
                    if not adv.get("summary") and not adv.get("description"):
                        continue
                    
                    title = adv.get("summary", "")[:200]
                    desc = adv.get("description", "")[:1000]
                    ghsa_id = adv.get("ghsa_id", "")
                    ecosystem = ""
                    
                    # Extract ecosystem from vulnerabilities
                    vulns = adv.get("vulnerabilities", [])
                    if vulns:
                        pkg = vulns[0].get("package", {})
                        ecosystem = pkg.get("ecosystem", "")
                    
                    if title and desc and ghsa_id:
                        entries.append({
                            "id": ghsa_id.lower().replace("-", ""),
                            "title": title,
                            "description": desc,
                            "label": "",
                            "predicted_label": "",
                            "source": f"https://github.com/advisories/{ghsa_id}",
                            "ecosystem": ecosystem,
                            "year": adv.get("published_at", "")[:4],
                            "raw_source": "GHSA"
                        })
            
            time.sleep(1)  # Rate limit
            
        except Exception as e:
            print(f"  Warning: {e}")
            continue
    
    print(f"  Fetched {len(entries)} GitHub advisories")
    return entries


def fetch_osv_advisories(max_entries=3000):
    """Fetch advisories from OSV.dev (Open Source Vulnerabilities database)."""
    print("\n[2/4] Fetching OSV.dev advisories...")
    entries = []
    
    # OSV ecosystems to query
    ecosystems = [
        "npm", "PyPI", "RubyGems", "crates.io", 
        "Maven", "Go", "Packagist", "NuGet"
    ]
    
    for ecosystem in ecosystems:
        try:
            url = "https://api.osv.dev/v1/query"
            payload = {
                "package": {"ecosystem": ecosystem},
                "page_size": 500
            }
            
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                vulns = data.get("vulns", [])
                
                for vuln in vulns:
                    vuln_id = vuln.get("id", "")
                    summary = vuln.get("summary", "")
                    details = vuln.get("details", "")
                    
                    # Filter for supply chain relevant entries
                    combined = (summary + " " + details).lower()
                    supply_chain_keywords = [
                        "malicious", "backdoor", "typosquat", "dependency confusion",
                        "supply chain", "pipeline", "postinstall", "setup.py",
                        "install hook", "unicode", "bidirectional"
                    ]
                    
                    if any(kw in combined for kw in supply_chain_keywords):
                        if summary and details and vuln_id:
                            # Get affected package info
                            affected = vuln.get("affected", [])
                            pkg_name = ""
                            if affected:
                                pkg = affected[0].get("package", {})
                                pkg_name = pkg.get("name", "")
                            
                            entries.append({
                                "id": vuln_id.lower().replace("-", "").replace(".", ""),
                                "title": f"{summary[:150]} ({pkg_name})" if pkg_name else summary[:150],
                                "description": details[:1000],
                                "label": "",
                                "predicted_label": "",
                                "source": f"https://osv.dev/vulnerability/{vuln_id}",
                                "ecosystem": ecosystem,
                                "year": vuln.get("published", "")[:4],
                                "raw_source": "OSV"
                            })
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Warning ({ecosystem}): {e}")
            continue
    
    print(f"  Fetched {len(entries)} OSV advisories")
    return entries


def fetch_cisa_kev():
    """Fetch supply chain entries from CISA Known Exploited Vulnerabilities."""
    print("\n[3/4] Fetching CISA KEV entries...")
    entries = []
    
    try:
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        resp = requests.get(url, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            
            supply_chain_keywords = [
                "supply chain", "package", "npm", "pypi", "dependency",
                "backdoor", "malicious", "pipeline", "ci/cd", "github actions"
            ]
            
            for vuln in vulns:
                desc = vuln.get("shortDescription", "")
                name = vuln.get("vulnerabilityName", "")
                combined = (desc + " " + name).lower()
                
                if any(kw in combined for kw in supply_chain_keywords):
                    cve_id = vuln.get("cveID", "")
                    entries.append({
                        "id": cve_id.lower().replace("-", ""),
                        "title": name[:200],
                        "description": desc[:1000],
                        "label": "",
                        "predicted_label": "",
                        "source": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        "ecosystem": "",
                        "year": vuln.get("dateAdded", "")[:4],
                        "raw_source": "CISA-KEV"
                    })
    
    except Exception as e:
        print(f"  Warning: {e}")
    
    print(f"  Fetched {len(entries)} CISA KEV entries")
    return entries


def fetch_sonatype_data():
    """Generate additional entries from known Sonatype documented incidents."""
    print("\n[4/4] Loading supplementary known incidents...")
    
    # High-quality known incidents not in original dataset
    # These are well-documented real incidents
    known_incidents = [
        {
            "id": "supp001",
            "title": "ctx and phpass PyPI packages compromised to steal environment variables",
            "description": "Attackers took over the abandoned PyPI packages ctx and phpass by registering the lapsed domain names of their maintainers, uploading new versions containing code that exfiltrated environment variables to an attacker-controlled server, affecting thousands of developers who had these packages as transitive dependencies.",
            "label": "AV-400",
            "source": "https://snyk.io/blog/pypi-malware-ctx-phpass/",
            "ecosystem": "PyPI", "year": "2022", "raw_source": "KNOWN"
        },
        {
            "id": "supp002",
            "title": "node-ipc sabotage attack targeting Russian and Belarusian IP addresses",
            "description": "The maintainer of the widely-used node-ipc npm package intentionally introduced destructive code in versions 10.1.1 and 10.1.2 that deleted files on systems with Russian or Belarusian IP addresses, affecting thousands of downstream projects including the Vue CLI.",
            "label": "AV-400",
            "source": "https://snyk.io/blog/peacenotwar-malicious-npm-node-ipc-package-vulnerability/",
            "ecosystem": "npm", "year": "2022", "raw_source": "KNOWN"
        },
        {
            "id": "supp003",
            "title": "Heroku and Travis CI OAuth token theft via stolen GitHub integration tokens",
            "description": "Attackers used stolen OAuth tokens issued to Heroku and Travis CI to download data from dozens of private GitHub repositories, including repositories belonging to npm. The tokens were not stolen from GitHub itself but from these third-party CI/CD services that had GitHub OAuth access.",
            "label": "AV-410",
            "source": "https://github.blog/2022-04-15-security-alert-stolen-oauth-user-tokens/",
            "ecosystem": "GitHub Actions", "year": "2022", "raw_source": "KNOWN"
        },
        {
            "id": "supp004",
            "title": "PyTorch-nightly dependency confusion via torchtriton package on PyPI",
            "description": "A malicious package named torchtriton was uploaded to PyPI, exploiting dependency confusion since PyTorch-nightly used an internal package of the same name. The malicious package contained a binary that collected system information and uploaded it to a remote server.",
            "label": "AV-509",
            "source": "https://pytorch.org/blog/compromised-nightly-dependency/",
            "ecosystem": "PyPI", "year": "2023", "raw_source": "KNOWN"
        },
        {
            "id": "supp005",
            "title": "LofyGang threat actor published 200+ malicious npm packages for credential theft",
            "description": "The LofyGang threat actor group published over 200 malicious npm packages impersonating legitimate packages including discord.js-selfbot and others, stealing Discord tokens, credit card information, and gaming platform credentials from developer machines.",
            "label": "AV-200",
            "source": "https://checkmarx.com/blog/lofygang-a-new-cybercrime-group-exploiting-open-source-ecosystem/",
            "ecosystem": "npm", "year": "2022", "raw_source": "KNOWN"
        },
        {
            "id": "supp006",
            "title": "3CX supply chain attack via compromised Electron application build",
            "description": "Nation-state attackers compromised the build environment of 3CX, a VoIP software company, injecting malicious code into the company's Electron-based desktop application. The compromised application was signed with 3CX's legitimate certificate and distributed through official channels to 600,000+ customers.",
            "label": "AV-400",
            "source": "https://www.mandiant.com/resources/blog/3cx-software-supply-chain-compromise",
            "ecosystem": "Other", "year": "2023", "raw_source": "KNOWN"
        },
        {
            "id": "supp007",
            "title": "Ultralytics PyPI package compromised via GitHub Actions workflow injection",
            "description": "Attackers exploited a GitHub Actions workflow vulnerability in the popular Ultralytics YOLO repository to inject malicious code into the PyPI release, deploying a cryptominer to systems that installed the compromised package versions 8.3.41 and 8.3.42.",
            "label": "AV-410",
            "source": "https://unit42.paloaltonetworks.com/ultralytics-supply-chain-attack/",
            "ecosystem": "PyPI", "year": "2024", "raw_source": "KNOWN"
        },
        {
            "id": "supp008",
            "title": "Backstabber campaign targeting npm with 200+ packages impersonating popular libraries",
            "description": "A coordinated campaign published over 200 malicious npm packages impersonating popular libraries including react, lodash, and express with slightly misspelled names, each containing obfuscated code that executed on installation to steal developer credentials and environment variables.",
            "label": "AV-200",
            "source": "https://socket.dev/blog/tag/supply-chain-attack",
            "ecosystem": "npm", "year": "2023", "raw_source": "KNOWN"
        },
    ]
    
    # Add predicted_label same as label for known high-confidence entries
    for entry in known_incidents:
        entry["predicted_label"] = entry["label"]
    
    print(f"  Loaded {len(known_incidents)} supplementary known incidents")
    return known_incidents


# ============================================================
# GPT-4 LABELING
# ============================================================

def label_with_gpt4(entries, api_key, batch_size=10):
    """Use GPT-4 to label unlabeled entries using taxonomy-aligned prompt."""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    unlabeled = [e for e in entries if not e.get("label")]
    print(f"\nLabeling {len(unlabeled)} unlabeled entries with GPT-4...")
    print(f"Estimated cost: ${len(unlabeled) * 0.001:.2f} - ${len(unlabeled) * 0.003:.2f}")
    
    labeled_count = 0
    error_count = 0
    
    for i, entry in enumerate(tqdm(unlabeled)):
        text = f"Title: {entry['title']}\n\nDescription: {entry['description']}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": TAXONOMY_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=10
            )
            
            label = response.choices[0].message.content.strip()
            
            # Validate label
            if label in VALID_LABELS:
                entry["label"] = label
                entry["predicted_label"] = label
                labeled_count += 1
            else:
                # Try to extract valid label from response
                for valid in VALID_LABELS:
                    if valid in label:
                        entry["label"] = valid
                        entry["predicted_label"] = valid
                        labeled_count += 1
                        break
                else:
                    entry["label"] = "UNCLASSIFIED"
                    error_count += 1
            
            # Rate limiting
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(unlabeled)} labeled, {error_count} errors")
                time.sleep(2)
            else:
                time.sleep(0.3)
                
        except Exception as e:
            print(f"  Error on entry {i}: {e}")
            entry["label"] = "ERROR"
            error_count += 1
            time.sleep(5)
    
    print(f"Labeling complete: {labeled_count} labeled, {error_count} errors")
    return entries


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(entries):
    """Remove duplicate entries based on title similarity and source URL."""
    print(f"\nDeduplicating {len(entries)} entries...")
    
    seen_ids = set()
    seen_sources = set()
    seen_title_hashes = set()
    unique = []
    
    for entry in entries:
        # Check by ID
        entry_id = entry.get("id", "")
        if entry_id and entry_id in seen_ids:
            continue
        
        # Check by source URL
        source = entry.get("source", "")
        if source and source in seen_sources:
            continue
        
        # Check by title hash (normalized)
        title = entry.get("title", "").lower().strip()
        title_hash = hashlib.md5(title.encode()).hexdigest()
        if title_hash in seen_title_hashes:
            continue
        
        # Check minimum quality
        if len(entry.get("description", "")) < 50:
            continue
        if not entry.get("title", "").strip():
            continue
        
        seen_ids.add(entry_id)
        if source:
            seen_sources.add(source)
        seen_title_hashes.add(title_hash)
        unique.append(entry)
    
    print(f"After deduplication: {len(unique)} unique entries")
    return unique


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Expand OSS threat dataset')
    parser.add_argument('--api-key', type=str, default='',
                       help='OpenAI API key for GPT-4 labeling')
    parser.add_argument('--input', type=str, 
                       default='oss_threat_dataset_beta.csv',
                       help='Path to existing dataset CSV')
    parser.add_argument('--output', type=str,
                       default='oss_threat_dataset_expanded.csv',
                       help='Output CSV path')
    parser.add_argument('--no-api', action='store_true',
                       help='Skip GPT-4 labeling (collect only)')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target dataset size')
    args = parser.parse_args()

    print("=" * 60)
    print("OSS THREAT DATASET EXPANSION PIPELINE")
    print(f"Target: {args.target} entries")
    print("=" * 60)

    # Step 1: Load existing dataset
    existing_entries = []
    try:
        df_existing = pd.read_csv(args.input)
        existing_entries = df_existing.to_dict('records')
        print(f"\nLoaded existing dataset: {len(existing_entries)} entries")
    except FileNotFoundError:
        print(f"\nWarning: {args.input} not found. Starting fresh.")

    # Step 2: Collect new entries from APIs
    new_entries = []
    new_entries.extend(fetch_github_advisories())
    new_entries.extend(fetch_osv_advisories())
    new_entries.extend(fetch_cisa_kev())
    new_entries.extend(fetch_sonatype_data())

    print(f"\nTotal new entries collected: {len(new_entries)}")

    # Step 3: Combine and deduplicate
    all_entries = existing_entries + new_entries
    all_entries = deduplicate(all_entries)

    print(f"Total unique entries after dedup: {len(all_entries)}")

    # Step 4: Label unlabeled entries with GPT-4
    if not args.no_api and args.api_key:
        all_entries = label_with_gpt4(all_entries, args.api_key)
    else:
        print("\nSkipping GPT-4 labeling (--no-api flag or no API key)")
        print("Run with --api-key YOUR_KEY to label entries")

    # Step 5: Filter to only labeled entries
    labeled = [e for e in all_entries if e.get("label") in VALID_LABELS]
    print(f"\nFinal labeled entries: {len(labeled)}")

    # Step 6: Show label distribution
    from collections import Counter
    dist = Counter(e["label"] for e in labeled)
    print("\nLabel distribution:")
    for label in VALID_LABELS:
        count = dist.get(label, 0)
        pct = count / len(labeled) * 100 if labeled else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Step 7: Save output
    df_out = pd.DataFrame(labeled)
    
    # Ensure correct column order
    cols = ["id", "title", "description", "label", 
            "predicted_label", "source", "ecosystem", "year"]
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out = df_out[cols]
    
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved {len(labeled)} entries to {args.output}")

    if len(labeled) < args.target:
        needed = args.target - len(labeled)
        print(f"\nNote: {needed} more entries needed to reach target of {args.target}")
        print("Run again after more advisories are published or broaden search terms")
    else:
        print(f"\n✅ Target of {args.target} entries reached!")

    print("\nDone.")


if __name__ == "__main__":
    main()
