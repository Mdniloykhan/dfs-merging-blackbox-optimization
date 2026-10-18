#!/usr/bin/env python3
# ============================================================
# AUTOMATED DATASET EXPANSION SCRIPT v2
# Grows OSS threat dataset from 999 to 10,000+ entries
# FIXES: OSV.dev API, more sources, better filtering
# ============================================================
# Usage:
#   pip install requests pandas openai tqdm
#   set GITHUB_TOKEN=your_token_here  (Windows)
#   python expand_dataset_v2.py --no-api --input oss_threat_dataset_beta.csv
#   python expand_dataset_v2.py --api-key YOUR_OPENAI_KEY --input oss_threat_dataset_beta.csv
# ============================================================

import requests
import pandas as pd
import json
import time
import argparse
import hashlib
import os
from tqdm import tqdm
from collections import Counter

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

TAXONOMY_PROMPT = """You are an expert in open source software supply chain security.
Classify the following security incident into exactly one of these categories:

AV-200 (Typosquatting): Malicious packages that mimic legitimate package names through character substitution, transposition, or namespace impersonation.

AV-300 (Trojan Source): Attacks exploiting Unicode bidirectional control characters or homoglyphs to create source code that executes differently from its visual representation.

AV-400 (Malicious Builds): Packages containing malicious code executed during installation, build, or import processes through lifecycle hooks such as postinstall and setup.py.

AV-410 (Pipeline Poisoning): Attacks that compromise CI/CD pipeline execution including GitHub Actions workflow injection, poisoned pipeline execution (PPE), and build system compromise.

AV-509 (Dependency Confusion): Attacks exploiting package manager resolution priority to cause build systems to install malicious public packages instead of intended private internal dependencies.

Respond with ONLY the category label (AV-200, AV-300, AV-400, AV-410, or AV-509). No explanation."""

SUPPLY_CHAIN_KEYWORDS = [
    "malicious", "backdoor", "typosquat", "dependency confusion",
    "supply chain", "pipeline", "postinstall", "setup.py",
    "install hook", "unicode", "bidirectional", "npm package",
    "pypi package", "malware", "trojan", "obfuscat",
    "credential steal", "exfiltrat", "cryptominer", "reverse shell",
    "environment variable", "token steal", "ci/cd", "github action",
    "workflow inject", "ppe attack", "package hijack", "repo hijack",
    "namespace confusion", "brandjacking", "impersonat"
]

def is_supply_chain_related(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in SUPPLY_CHAIN_KEYWORDS)

# ============================================================
# SOURCE 1: GITHUB SECURITY ADVISORIES
# ============================================================

def fetch_github_advisories():
    print("\n[1/6] Fetching GitHub Security Advisories...")
    entries = []
    headers = {"Accept": "application/vnd.github+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
        print(f"  Using GitHub token: {gh_token[:8]}...")
    else:
        print("  WARNING: No GitHub token. Results will be limited.")

    # Fetch malware advisories
    for type_ in ["malware", "reviewed"]:
        page = 1
        while page <= 10:
            url = f"https://api.github.com/advisories?type={type_}&per_page=100&page={page}"
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    advisories = resp.json()
                    if not advisories:
                        break
                    for adv in advisories:
                        summary = adv.get("summary", "")
                        description = adv.get("description", "")
                        ghsa_id = adv.get("ghsa_id", "")
                        if not (summary and description and ghsa_id):
                            continue
                        combined = summary + " " + description
                        if not is_supply_chain_related(combined):
                            continue
                        ecosystem = ""
                        vulns = adv.get("vulnerabilities", [])
                        if vulns:
                            pkg = vulns[0].get("package", {})
                            ecosystem = pkg.get("ecosystem", "")
                        entries.append({
                            "id": ghsa_id.lower().replace("-", ""),
                            "title": summary[:200],
                            "description": description[:1000],
                            "label": "",
                            "predicted_label": "",
                            "source": f"https://github.com/advisories/{ghsa_id}",
                            "ecosystem": ecosystem,
                            "year": adv.get("published_at", "")[:4],
                            "raw_source": "GHSA"
                        })
                    page += 1
                    time.sleep(0.5)
                elif resp.status_code == 403:
                    print(f"  Rate limited on page {page}. Waiting 60s...")
                    time.sleep(60)
                else:
                    break
            except Exception as e:
                print(f"  Warning: {e}")
                break

    print(f"  Fetched {len(entries)} GitHub advisories")
    return entries

# ============================================================
# SOURCE 2: OSV.DEV API (FIXED)
# ============================================================

def fetch_osv_advisories():
    print("\n[2/6] Fetching OSV.dev advisories (fixed)...")
    entries = []

    ecosystems = ["npm", "PyPI", "RubyGems", "crates.io", "Maven", "Go"]

    for ecosystem in ecosystems:
        print(f"  Querying {ecosystem}...")
        page_token = None
        page_count = 0

        while page_count < 20:
            try:
                url = "https://api.osv.dev/v1/query"
                payload = {
                    "package": {"ecosystem": ecosystem},
                    "page_size": 1000
                }
                if page_token:
                    payload["page_token"] = page_token

                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )

                if resp.status_code != 200:
                    print(f"    Error {resp.status_code} for {ecosystem}")
                    break

                data = resp.json()
                vulns = data.get("vulns", [])

                if not vulns:
                    break

                for vuln in vulns:
                    vuln_id = vuln.get("id", "")
                    summary = vuln.get("summary", "")
                    details = vuln.get("details", "")

                    if not summary and not details:
                        continue

                    combined = (summary + " " + details)
                    if not is_supply_chain_related(combined):
                        continue

                    # Get package name
                    affected = vuln.get("affected", [])
                    pkg_name = ""
                    if affected:
                        pkg = affected[0].get("package", {})
                        pkg_name = pkg.get("name", "")

                    title = f"{summary[:150]}"
                    if pkg_name:
                        title = f"{pkg_name}: {summary[:120]}"

                    entries.append({
                        "id": vuln_id.lower().replace("-", "").replace(".", ""),
                        "title": title[:200],
                        "description": details[:1000] if details else summary[:1000],
                        "label": "",
                        "predicted_label": "",
                        "source": f"https://osv.dev/vulnerability/{vuln_id}",
                        "ecosystem": ecosystem,
                        "year": vuln.get("published", "")[:4],
                        "raw_source": "OSV"
                    })

                page_token = data.get("next_page_token")
                if not page_token:
                    break

                page_count += 1
                time.sleep(1)

            except Exception as e:
                print(f"    Warning ({ecosystem}): {e}")
                break

        print(f"    {ecosystem}: found relevant entries so far: {len([e for e in entries if e.get('ecosystem') == ecosystem])}")

    print(f"  Fetched {len(entries)} OSV advisories")
    return entries

# ============================================================
# SOURCE 3: CISA KEV (IMPROVED)
# ============================================================

def fetch_cisa_kev():
    print("\n[3/6] Fetching CISA KEV entries...")
    entries = []
    try:
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            for vuln in vulns:
                desc = vuln.get("shortDescription", "")
                name = vuln.get("vulnerabilityName", "")
                combined = desc + " " + name
                if is_supply_chain_related(combined):
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

# ============================================================
# SOURCE 4: GITHUB GHSL ADVISORIES (SECURITY LAB)
# ============================================================

def fetch_ghsl_advisories():
    print("\n[4/6] Fetching GitHub Security Lab (GHSL) advisories...")
    entries = []
    headers = {"Accept": "application/vnd.github+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    # Search for PPE and supply chain specific issues
    queries = [
        "poisoned pipeline execution",
        "workflow injection supply chain",
        "github actions malicious",
        "ci cd supply chain attack",
        "typosquatting npm pypi",
        "dependency confusion attack",
        "malicious package npm",
        "malicious package pypi"
    ]

    for query in queries:
        try:
            url = f"https://api.github.com/search/repositories?q={query.replace(' ', '+')}+topic:security&per_page=30"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                for item in items:
                    desc = item.get("description", "")
                    name = item.get("full_name", "")
                    if desc and is_supply_chain_related(desc + " " + name):
                        entries.append({
                            "id": f"ghrepo{hashlib.md5(name.encode()).hexdigest()[:8]}",
                            "title": f"{name}: {desc[:150]}",
                            "description": desc[:500],
                            "label": "",
                            "predicted_label": "",
                            "source": item.get("html_url", ""),
                            "ecosystem": "",
                            "year": item.get("created_at", "")[:4],
                            "raw_source": "GHSL"
                        })
            time.sleep(2)
        except Exception as e:
            print(f"  Warning: {e}")
            continue

    print(f"  Fetched {len(entries)} GHSL entries")
    return entries

# ============================================================
# SOURCE 5: NVD CVE API
# ============================================================

def fetch_nvd_cves():
    print("\n[5/6] Fetching NVD CVE entries...")
    entries = []

    keywords = [
        "supply chain attack",
        "malicious npm package",
        "malicious pypi package",
        "dependency confusion",
        "typosquatting package",
        "github actions compromise",
        "ci cd pipeline attack"
    ]

    for keyword in keywords:
        try:
            url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keyword.replace(' ', '%20')}&resultsPerPage=50"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                vulns = data.get("vulnerabilities", [])
                for item in vulns:
                    cve = item.get("cve", {})
                    cve_id = cve.get("id", "")
                    descriptions = cve.get("descriptions", [])
                    desc = ""
                    for d in descriptions:
                        if d.get("lang") == "en":
                            desc = d.get("value", "")
                            break
                    if desc and is_supply_chain_related(desc):
                        entries.append({
                            "id": cve_id.lower().replace("-", ""),
                            "title": f"{cve_id}: {desc[:150]}",
                            "description": desc[:1000],
                            "label": "",
                            "predicted_label": "",
                            "source": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                            "ecosystem": "",
                            "year": cve.get("published", "")[:4],
                            "raw_source": "NVD"
                        })
            time.sleep(3)  # NVD rate limit
        except Exception as e:
            print(f"  Warning: {e}")
            continue

    print(f"  Fetched {len(entries)} NVD CVE entries")
    return entries

# ============================================================
# SOURCE 6: SUPPLEMENTARY HIGH QUALITY KNOWN INCIDENTS
# ============================================================

def fetch_supplementary():
    print("\n[6/6] Loading supplementary known incidents...")
    incidents = [
        {"id": "supp001", "title": "ctx and phpass PyPI packages compromised to steal environment variables", "description": "Attackers took over the abandoned PyPI packages ctx and phpass by registering the lapsed domain names of their maintainers, uploading new versions containing code that exfiltrated environment variables to an attacker-controlled server, affecting thousands of developers who had these packages as transitive dependencies.", "label": "AV-400", "source": "https://snyk.io/blog/pypi-malware-ctx-phpass/", "ecosystem": "PyPI", "year": "2022", "raw_source": "KNOWN"},
        {"id": "supp002", "title": "node-ipc sabotage attack targeting Russian and Belarusian IP addresses", "description": "The maintainer of the widely-used node-ipc npm package intentionally introduced destructive code in versions 10.1.1 and 10.1.2 that deleted files on systems with Russian or Belarusian IP addresses, affecting thousands of downstream projects including the Vue CLI.", "label": "AV-400", "source": "https://snyk.io/blog/peacenotwar-malicious-npm-node-ipc-package-vulnerability/", "ecosystem": "npm", "year": "2022", "raw_source": "KNOWN"},
        {"id": "supp003", "title": "Heroku and Travis CI OAuth token theft via stolen GitHub integration tokens", "description": "Attackers used stolen OAuth tokens issued to Heroku and Travis CI to download data from dozens of private GitHub repositories. The tokens were not stolen from GitHub itself but from third-party CI/CD services that had GitHub OAuth access.", "label": "AV-410", "source": "https://github.blog/2022-04-15-security-alert-stolen-oauth-user-tokens/", "ecosystem": "GitHub Actions", "year": "2022", "raw_source": "KNOWN"},
        {"id": "supp004", "title": "PyTorch-nightly dependency confusion via torchtriton package on PyPI", "description": "A malicious package named torchtriton was uploaded to PyPI, exploiting dependency confusion since PyTorch-nightly used an internal package of the same name. The malicious package collected system information and uploaded it to a remote server.", "label": "AV-509", "source": "https://pytorch.org/blog/compromised-nightly-dependency/", "ecosystem": "PyPI", "year": "2023", "raw_source": "KNOWN"},
        {"id": "supp005", "title": "LofyGang published 200+ malicious npm packages for credential theft", "description": "The LofyGang threat actor group published over 200 malicious npm packages impersonating legitimate packages including discord.js-selfbot and others, stealing Discord tokens, credit card information, and gaming platform credentials.", "label": "AV-200", "source": "https://checkmarx.com/blog/lofygang-a-new-cybercrime-group-exploiting-open-source-ecosystem/", "ecosystem": "npm", "year": "2022", "raw_source": "KNOWN"},
        {"id": "supp006", "title": "3CX supply chain attack via compromised Electron application build", "description": "Nation-state attackers compromised the build environment of 3CX, injecting malicious code into the Electron-based desktop application. The compromised application was signed with 3CX's legitimate certificate and distributed to 600,000+ customers.", "label": "AV-400", "source": "https://www.mandiant.com/resources/blog/3cx-software-supply-chain-compromise", "ecosystem": "Other", "year": "2023", "raw_source": "KNOWN"},
        {"id": "supp007", "title": "Ultralytics PyPI package compromised via GitHub Actions workflow injection", "description": "Attackers exploited a GitHub Actions workflow vulnerability in the Ultralytics YOLO repository to inject malicious code into the PyPI release, deploying a cryptominer to systems that installed versions 8.3.41 and 8.3.42.", "label": "AV-410", "source": "https://unit42.paloaltonetworks.com/ultralytics-supply-chain-attack/", "ecosystem": "PyPI", "year": "2024", "raw_source": "KNOWN"},
        {"id": "supp008", "title": "Backstabber campaign targeting npm with 200+ impersonating packages", "description": "A coordinated campaign published over 200 malicious npm packages impersonating popular libraries including react, lodash, and express with slightly misspelled names, executing on installation to steal developer credentials.", "label": "AV-200", "source": "https://socket.dev/blog/tag/supply-chain-attack", "ecosystem": "npm", "year": "2023", "raw_source": "KNOWN"},
        {"id": "supp009", "title": "SolarWinds Orion SUNBURST backdoor injected via compromised build pipeline", "description": "Attackers compromised SolarWinds build infrastructure to inject the SUNBURST backdoor into legitimate Orion software updates, affecting 18,000+ organizations including US government agencies.", "label": "AV-400", "source": "https://cloud.google.com/blog/topics/threat-intelligence/evasive-attacker-leverages-solarwinds-supply-chain-compromises-with-sunburst-backdoor", "ecosystem": "Other", "year": "2020", "raw_source": "KNOWN"},
        {"id": "supp010", "title": "event-stream npm package backdoor targeting Copay Bitcoin wallet", "description": "A malicious maintainer added the flatmap-stream package as a dependency to event-stream, which contained encrypted malicious code targeting the Copay Bitcoin wallet application to steal private keys.", "label": "AV-400", "source": "https://blog.npmjs.org/post/180565383195/details-about-the-event-stream-incident", "ecosystem": "npm", "year": "2018", "raw_source": "KNOWN"},
        {"id": "supp011", "title": "XZ Utils backdoor CVE-2024-3094 via social engineering maintainer", "description": "A threat actor spent two years as a trusted contributor to XZ Utils before injecting a sophisticated backdoor in versions 5.6.0 and 5.6.1 that would have allowed unauthorized SSH access to affected Linux systems.", "label": "AV-400", "source": "https://nvd.nist.gov/vuln/detail/CVE-2024-3094", "ecosystem": "Other", "year": "2024", "raw_source": "KNOWN"},
        {"id": "supp012", "title": "tj-actions changed-files GitHub Action version tag tampering", "description": "Attackers compromised the tj-actions/changed-files GitHub Action by modifying version tags to point to a malicious commit that extracted CI/CD secrets from Runner Worker memory, affecting over 23,000 repositories.", "label": "AV-410", "source": "https://unit42.paloaltonetworks.com/github-actions-supply-chain-attack/", "ecosystem": "GitHub Actions", "year": "2025", "raw_source": "KNOWN"},
        {"id": "supp013", "title": "colourama PyPI typosquat of colorama with clipboard hijacker", "description": "A malicious PyPI package named colourama mimicked the popular colorama library through a one-character substitution, executing a cryptocurrency clipboard hijacker on Windows systems upon import.", "label": "AV-200", "source": "https://bertusk.medium.com/cryptocurrency-clipboard-hijacker-discovered-in-pypi-repository-b66b8a534a8", "ecosystem": "PyPI", "year": "2018", "raw_source": "KNOWN"},
        {"id": "supp014", "title": "twilio-npm brandjacking attack delivering reverse shell payload", "description": "The malicious npm package twilio-npm impersonated the official Twilio client library delivering a reverse shell payload upon installation, providing attackers persistent access to developer machines.", "label": "AV-200", "source": "https://github.com/advisories/GHSA-r558-h2gm-7q4g", "ecosystem": "npm", "year": "2020", "raw_source": "KNOWN"},
        {"id": "supp015", "title": "Trojan Source CVE-2021-42574 Unicode bidirectional attack in source code", "description": "Researchers demonstrated that invisible Unicode bidirectional control characters embedded in source code comments and strings cause compilers for C, C++, Go, Java, Python, and Rust to execute logic different from what developers see in their editors.", "label": "AV-300", "source": "https://trojansource.codes", "ecosystem": "Other", "year": "2021", "raw_source": "KNOWN"},
        {"id": "supp016", "title": "Alex Birsan dependency confusion research affecting 35+ major organizations", "description": "Security researcher Alex Birsan demonstrated dependency confusion by uploading packages with the same names as internal libraries used by Apple, Microsoft, Netflix, Tesla, and 32 other major organizations to public registries, causing their build systems to execute malicious code.", "label": "AV-509", "source": "https://medium.com/@alex.birsan/dependency-confusion-4a5d60fec610", "ecosystem": "npm", "year": "2021", "raw_source": "KNOWN"},
        {"id": "supp017", "title": "Codecov bash uploader tampered in supply chain attack exfiltrating CI secrets", "description": "Attackers modified the Codecov bash uploader script distributed from Codecov servers to include code that exfiltrated environment variables including CI/CD secrets from users running the script in their pipelines.", "label": "AV-410", "source": "https://about.codecov.io/security-update/", "ecosystem": "GitHub Actions", "year": "2021", "raw_source": "KNOWN"},
        {"id": "supp018", "title": "TeamTNT cryptocurrency mining campaign via malicious Docker Hub images", "description": "The TeamTNT threat actor group published malicious Docker images to Docker Hub that deployed cryptocurrency miners and credential stealers when pulled and executed by developers and CI/CD systems.", "label": "AV-200", "source": "https://unit42.paloaltonetworks.com/waterbear-teamtnt/", "ecosystem": "Docker", "year": "2021", "raw_source": "KNOWN"},
        {"id": "supp019", "title": "Protestware colors and faker npm packages broken by maintainer", "description": "The maintainer of the widely-used colors and faker npm packages intentionally broke the packages by introducing infinite loops and nonsensical output, affecting thousands of downstream projects that depended on these libraries.", "label": "AV-400", "source": "https://snyk.io/blog/open-source-npm-packages-colors-faker/", "ecosystem": "npm", "year": "2022", "raw_source": "KNOWN"},
        {"id": "supp020", "title": "Lazarus Group targeting software developers with malicious npm packages", "description": "North Korean state-sponsored Lazarus Group published malicious npm packages impersonating cryptocurrency-related tools to target blockchain developers, deploying remote access trojans that provided persistent access to development environments.", "label": "AV-200", "source": "https://unit42.paloaltonetworks.com/threat-assessment-north-korean-threat-groups/", "ecosystem": "npm", "year": "2023", "raw_source": "KNOWN"},
    ]

    for entry in incidents:
        entry["predicted_label"] = entry["label"]

    print(f"  Loaded {len(incidents)} supplementary incidents")
    return incidents

# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(entries):
    print(f"\nDeduplicating {len(entries)} entries...")
    seen_ids = set()
    seen_sources = set()
    seen_title_hashes = set()
    unique = []

    for entry in entries:
        entry_id = str(entry.get("id", "")).strip()
        source = str(entry.get("source", "")).strip()
        title = str(entry.get("title", "")).lower().strip()
        title_hash = hashlib.md5(title.encode()).hexdigest()

        if entry_id and entry_id in seen_ids:
            continue
        if source and source in seen_sources:
            continue
        if title_hash in seen_title_hashes:
            continue
        if len(str(entry.get("description", ""))) < 30:
            continue
        if not str(entry.get("title", "")).strip():
            continue

        if entry_id:
            seen_ids.add(entry_id)
        if source:
            seen_sources.add(source)
        seen_title_hashes.add(title_hash)
        unique.append(entry)

    print(f"After deduplication: {len(unique)} unique entries")
    return unique

# ============================================================
# GPT-4 LABELING
# ============================================================

def label_with_gpt4(entries, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)

    unlabeled = [e for e in entries if not e.get("label") or e["label"] == ""]
    print(f"\nLabeling {len(unlabeled)} unlabeled entries with GPT-4...")
    estimated_cost = len(unlabeled) * 0.002
    print(f"Estimated cost: ${estimated_cost:.2f}")

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

            if label in VALID_LABELS:
                entry["label"] = label
                entry["predicted_label"] = label
            else:
                for valid in VALID_LABELS:
                    if valid in label:
                        entry["label"] = valid
                        entry["predicted_label"] = valid
                        break
                else:
                    entry["label"] = "UNCLASSIFIED"
                    error_count += 1

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(unlabeled)}, errors: {error_count}")
            time.sleep(0.3)

        except Exception as e:
            print(f"  Error on entry {i}: {e}")
            entry["label"] = "ERROR"
            error_count += 1
            time.sleep(5)

    print(f"Labeling complete. Errors: {error_count}")
    return entries

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, default='')
    parser.add_argument('--input', type=str, default='oss_threat_dataset_beta.csv')
    parser.add_argument('--output', type=str, default='oss_threat_dataset_v2.csv')
    parser.add_argument('--no-api', action='store_true')
    parser.add_argument('--target', type=int, default=10000)
    args = parser.parse_args()

    print("=" * 60)
    print("OSS THREAT DATASET EXPANSION PIPELINE v2")
    print(f"Target: {args.target} entries")
    print("=" * 60)

    # Load existing
    existing = []
    try:
        df = pd.read_csv(args.input)
        existing = df.to_dict('records')
        print(f"\nLoaded existing dataset: {len(existing)} entries")
    except FileNotFoundError:
        print(f"\nWarning: {args.input} not found. Starting fresh.")

    # Collect from all sources
    new_entries = []
    new_entries.extend(fetch_github_advisories())
    new_entries.extend(fetch_osv_advisories())
    new_entries.extend(fetch_cisa_kev())
    new_entries.extend(fetch_ghsl_advisories())
    new_entries.extend(fetch_nvd_cves())
    new_entries.extend(fetch_supplementary())

    print(f"\nTotal new entries collected: {len(new_entries)}")

    # Combine and deduplicate
    all_entries = existing + new_entries
    all_entries = deduplicate(all_entries)
    print(f"Total unique entries: {len(all_entries)}")

    # Label with GPT-4
    if not args.no_api and args.api_key:
        all_entries = label_with_gpt4(all_entries, args.api_key)
    else:
        print("\nSkipping GPT-4 labeling. Run with --api-key to label.")

    # Filter labeled only
    labeled = [e for e in all_entries if e.get("label") in VALID_LABELS]
    print(f"\nFinal labeled entries: {len(labeled)}")

    # Distribution
    dist = Counter(e["label"] for e in labeled)
    print("\nLabel distribution:")
    for label in VALID_LABELS:
        count = dist.get(label, 0)
        pct = count / len(labeled) * 100 if labeled else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Save
    cols = ["id", "title", "description", "label",
            "predicted_label", "source", "ecosystem", "year"]
    df_out = pd.DataFrame(labeled)
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out = df_out[cols]
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved {len(labeled)} entries to {args.output}")

    needed = args.target - len(labeled)
    if needed > 0:
        print(f"\nNote: {needed} more entries needed to reach {args.target}")
    else:
        print(f"\n✅ Target of {args.target} entries reached!")

    print("Done.")

if __name__ == "__main__":
    main()
