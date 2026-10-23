#!/usr/bin/env python3
# ============================================================
# SECURITY BLOG SCRAPER
# Scrapes Snyk, Socket.dev, and Checkmarx for supply chain
# incident reports not covered by GitHub Advisory Database
# ============================================================
# Usage:
#   pip install requests pandas tqdm beautifulsoup4
#   python scrape_security_blogs.py --input oss_threat_dataset_10k.csv
# Output: oss_threat_dataset_scraped.csv
# ============================================================

import requests
import pandas as pd
import hashlib
import time
import argparse
import re
import os
from tqdm import tqdm
from collections import Counter
from bs4 import BeautifulSoup

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

SUPPLY_CHAIN_KEYWORDS = [
    "malicious", "backdoor", "typosquat", "dependency confusion",
    "supply chain", "pipeline", "postinstall", "setup.py",
    "unicode bidirectional", "trojan source", "malware", "trojan",
    "credential steal", "exfiltrat", "cryptominer", "reverse shell",
    "ci/cd", "github action", "workflow inject", "package hijack",
    "namespace confusion", "brandjack", "impersonat",
    "fake package", "malicious package", "install hook",
    "lifecycle hook", "dropper", "obfuscat", "token steal",
    "secret steal", "environment variable"
]

def is_supply_chain(text):
    t = text.lower()
    return any(kw in t for kw in SUPPLY_CHAIN_KEYWORDS)

def classify_rule_based(title, description):
    text = (title + " " + description).lower()

    if any(kw in text for kw in [
        "unicode bidirectional","bidi character","trojan source",
        "invisible character","zero-width","homoglyph",
        "bidirectional control","rtlo","right-to-left override"]):
        return "AV-300", 0.95

    if any(kw in text for kw in [
        "dependency confusion","namespace confusion",
        "internal package","private package","public registry override",
        "registry priority","internal dependency","private dependency",
        "confused deputy"]):
        return "AV-509", 0.90

    if any(kw in text for kw in [
        "github actions","workflow inject","poisoned pipeline",
        "ppe attack","pipeline poison","ci/cd compromise",
        "workflow file","runner worker","actions secret",
        "pipeline execution","workflow token","pull request injection",
        "cicd secret","build pipeline compromis"]):
        return "AV-410", 0.90

    if any(kw in text for kw in [
        "typosquat","typo-squat","misspell","impersonat",
        "mimic","look-alike","lookalike","character substitut",
        "brandjack","name squatt","one character","one-character",
        "namespace impersonat","fake package","package impersonat",
        "copied package","similar name"]):
        return "AV-200", 0.88

    if any(kw in text for kw in [
        "postinstall","setup.py","install script","lifecycle hook",
        "malicious code execut","code execut on install",
        "backdoor inject","reverse shell","cryptominer",
        "credential steal","exfiltrat","malicious payload",
        "dropper","malicious npm","malicious pypi",
        "malicious package","install malicious"]):
        return "AV-400", 0.85

    if any(kw in text for kw in ["ci/cd","pipeline","workflow"]):
        return "AV-410", 0.72
    if any(kw in text for kw in ["malicious","backdoor","trojan","malware"]):
        return "AV-400", 0.72

    return "UNCERTAIN", 0.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ============================================================
# SOURCE 1: SNYK VULNERABILITY DATABASE
# ============================================================

def scrape_snyk():
    print("\n[1/3] Scraping Snyk Vulnerability Database...")
    entries = []

    # Snyk has a public vuln DB with npm/PyPI malicious packages
    base_urls = [
        "https://security.snyk.io/vuln/npm?type=malicious-code&page={}",
        "https://security.snyk.io/vuln/pip?type=malicious-code&page={}",
        "https://security.snyk.io/vuln/rubygems?type=malicious-code&page={}",
        "https://security.snyk.io/vuln/maven?type=malicious-code&page={}",
    ]

    ecosystems = ["npm", "PyPI", "RubyGems", "Maven"]

    for url_template, ecosystem in zip(base_urls, ecosystems):
        print(f"  Scraping Snyk {ecosystem}...")
        page = 1
        empty_pages = 0

        while page <= 50 and empty_pages < 3:
            url = url_template.format(page)
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code != 200:
                    print(f"    Status {resp.status_code} on page {page}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Find vulnerability cards
                vuln_items = soup.find_all('a', class_=re.compile(r'vue--card|vuln-item|vulnerability'))
                if not vuln_items:
                    # Try alternative selectors
                    vuln_items = soup.find_all('div', class_=re.compile(r'vue--card|vuln'))

                if not vuln_items:
                    empty_pages += 1
                    page += 1
                    time.sleep(1)
                    continue

                found = 0
                for item in vuln_items:
                    try:
                        # Extract title
                        title_el = item.find(['h3', 'h4', 'strong', 'span'],
                                            class_=re.compile(r'title|name|vuln'))
                        title = title_el.get_text(strip=True) if title_el else ""

                        # Extract description
                        desc_el = item.find(['p', 'div'],
                                           class_=re.compile(r'desc|detail|summary'))
                        desc = desc_el.get_text(strip=True) if desc_el else ""

                        # Extract link
                        href = item.get('href', '')
                        if not href and item.name != 'a':
                            link_el = item.find('a')
                            href = link_el.get('href', '') if link_el else ''

                        if not title:
                            continue

                        combined = title + " " + desc
                        if not is_supply_chain(combined):
                            continue

                        vuln_id = href.split('/')[-1] if href else ""
                        source = f"https://security.snyk.io{href}" if href.startswith('/') else href

                        entry_id = f"snyk{hashlib.md5((title+ecosystem).encode()).hexdigest()[:10]}"

                        entries.append({
                            "id": entry_id,
                            "title": title[:200],
                            "description": desc[:1000] if desc else title[:500],
                            "label": "",
                            "predicted_label": "",
                            "source": source or f"https://security.snyk.io/vuln/{ecosystem.lower()}",
                            "ecosystem": ecosystem,
                            "year": "",
                            "raw_source": "SNYK"
                        })
                        found += 1

                    except Exception:
                        continue

                if found == 0:
                    empty_pages += 1
                else:
                    empty_pages = 0

                page += 1
                time.sleep(1.5)

            except Exception as e:
                print(f"    Warning page {page}: {e}")
                page += 1
                time.sleep(3)

        print(f"    {ecosystem}: collected so far {len([e for e in entries if e.get('ecosystem')==ecosystem])}")

    print(f"  Snyk total: {len(entries)}")
    return entries


# ============================================================
# SOURCE 2: SOCKET.DEV BLOG
# ============================================================

def scrape_socket():
    print("\n[2/3] Scraping Socket.dev security reports...")
    entries = []

    # Socket publishes individual package threat reports
    blog_pages = [
        "https://socket.dev/blog",
        "https://socket.dev/blog/page/2",
        "https://socket.dev/blog/page/3",
        "https://socket.dev/blog/page/4",
        "https://socket.dev/blog/page/5",
    ]

    article_urls = []

    # First collect article URLs from blog pages
    for blog_url in blog_pages:
        try:
            resp = requests.get(blog_url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find article links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/blog/' in href and len(href) > 20:
                    if href.startswith('/'):
                        href = f"https://socket.dev{href}"
                    if href not in article_urls and 'socket.dev/blog' in href:
                        article_urls.append(href)

            time.sleep(1)
        except Exception as e:
            print(f"  Warning: {e}")

    print(f"  Found {len(article_urls)} Socket.dev articles")

    # Scrape each article
    for url in tqdm(article_urls[:100], desc="  Scraping articles"):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Get title
            title_el = soup.find(['h1', 'h2'])
            title = title_el.get_text(strip=True) if title_el else ""

            # Get description from meta or first paragraph
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                desc = meta_desc.get('content', '')
            else:
                paras = soup.find_all('p')
                desc = ' '.join(p.get_text(strip=True) for p in paras[:3])

            if not title:
                continue

            combined = title + " " + desc
            if not is_supply_chain(combined):
                continue

            # Try to extract individual package mentions
            # Socket often lists multiple packages per article
            pkg_mentions = re.findall(
                r'package[s]?\s+(?:named?|called?|like?|including?)\s+[`\'"]([^`\'"]+)[`\'"]',
                soup.get_text(), re.IGNORECASE
            )

            if pkg_mentions:
                # Create entry per package
                for pkg in pkg_mentions[:10]:
                    entry_id = f"socket{hashlib.md5((pkg+url).encode()).hexdigest()[:10]}"
                    entries.append({
                        "id": entry_id,
                        "title": f"{pkg}: {title[:150]}",
                        "description": desc[:1000],
                        "label": "",
                        "predicted_label": "",
                        "source": url,
                        "ecosystem": "npm" if "npm" in combined.lower() else
                                    "PyPI" if "pypi" in combined.lower() else "",
                        "year": "",
                        "raw_source": "SOCKET"
                    })
            else:
                # Single entry for the article
                entry_id = f"socket{hashlib.md5(url.encode()).hexdigest()[:10]}"
                entries.append({
                    "id": entry_id,
                    "title": title[:200],
                    "description": desc[:1000],
                    "label": "",
                    "predicted_label": "",
                    "source": url,
                    "ecosystem": "npm" if "npm" in combined.lower() else
                                "PyPI" if "pypi" in combined.lower() else "",
                    "year": "",
                    "raw_source": "SOCKET"
                })

            time.sleep(1)

        except Exception as e:
            continue

    print(f"  Socket.dev total: {len(entries)}")
    return entries


# ============================================================
# SOURCE 3: CHECKMARX SUPPLY CHAIN RESEARCH
# ============================================================

def scrape_checkmarx():
    print("\n[3/3] Scraping Checkmarx supply chain research...")
    entries = []

    blog_pages = [
        "https://checkmarx.com/blog/category/software-supply-chain-security/",
        "https://checkmarx.com/blog/category/software-supply-chain-security/page/2/",
        "https://checkmarx.com/blog/category/software-supply-chain-security/page/3/",
        "https://checkmarx.com/blog/category/software-supply-chain-security/page/4/",
        "https://checkmarx.com/blog/category/software-supply-chain-security/page/5/",
    ]

    article_urls = []

    for blog_url in blog_pages:
        try:
            resp = requests.get(blog_url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'checkmarx.com/blog' in href and len(href) > 40:
                    if href not in article_urls:
                        article_urls.append(href)

            time.sleep(1)
        except Exception as e:
            print(f"  Warning: {e}")

    print(f"  Found {len(article_urls)} Checkmarx articles")

    for url in tqdm(article_urls[:80], desc="  Scraping articles"):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            title_el = soup.find('h1')
            title = title_el.get_text(strip=True) if title_el else ""

            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                desc = meta_desc.get('content', '')
            else:
                paras = soup.find_all('p')
                desc = ' '.join(p.get_text(strip=True) for p in paras[:4])

            if not title:
                continue

            combined = title + " " + desc
            if not is_supply_chain(combined):
                continue

            # Extract package names from article
            pkg_mentions = re.findall(
                r'[`\'"]([a-zA-Z0-9_\-\.]+)[`\'"]',
                soup.get_text()
            )
            pkg_mentions = [p for p in pkg_mentions
                          if 3 < len(p) < 50 and not p.startswith('http')][:5]

            if pkg_mentions:
                for pkg in pkg_mentions:
                    entry_id = f"cmx{hashlib.md5((pkg+url).encode()).hexdigest()[:10]}"
                    entries.append({
                        "id": entry_id,
                        "title": f"{pkg}: {title[:150]}",
                        "description": desc[:1000],
                        "label": "",
                        "predicted_label": "",
                        "source": url,
                        "ecosystem": "npm" if "npm" in combined.lower() else
                                    "PyPI" if "pypi" in combined.lower() else "",
                        "year": "",
                        "raw_source": "CHECKMARX"
                    })
            else:
                entry_id = f"cmx{hashlib.md5(url.encode()).hexdigest()[:10]}"
                entries.append({
                    "id": entry_id,
                    "title": title[:200],
                    "description": desc[:1000],
                    "label": "",
                    "predicted_label": "",
                    "source": url,
                    "ecosystem": "npm" if "npm" in combined.lower() else
                                "PyPI" if "pypi" in combined.lower() else "",
                    "year": "",
                    "raw_source": "CHECKMARX"
                })

            time.sleep(1.5)

        except Exception:
            continue

    print(f"  Checkmarx total: {len(entries)}")
    return entries


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(entries):
    print(f"\nDeduplicating {len(entries)} entries...")
    seen_ids = set()
    seen_sources = set()
    seen_hashes = set()
    unique = []

    for entry in entries:
        eid = str(entry.get("id", "")).strip()
        src = str(entry.get("source", "")).strip()
        title = str(entry.get("title", "")).lower().strip()
        thash = hashlib.md5(title.encode()).hexdigest()

        if eid and eid in seen_ids: continue
        if src and src in seen_sources: continue
        if thash in seen_hashes: continue
        if len(str(entry.get("description", ""))) < 20: continue
        if not title: continue

        if eid: seen_ids.add(eid)
        if src: seen_sources.add(src)
        seen_hashes.add(thash)
        unique.append(entry)

    print(f"After dedup: {len(unique)} unique entries")
    return unique


# ============================================================
# LABELING
# ============================================================

def label_entries(entries):
    print(f"\nLabeling {len(entries)} entries...")
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

        if label != "UNCERTAIN" and conf >= 0.85:
            entry["label"] = label
            entry["predicted_label"] = label
            labeled.append(entry)
        else:
            entry["label"] = "REVIEW_NEEDED"
            uncertain.append(entry)

    return labeled, uncertain


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='oss_threat_dataset_10k.csv')
    parser.add_argument('--output', default='oss_threat_dataset_scraped.csv')
    args = parser.parse_args()

    print("=" * 60)
    print("SECURITY BLOG SCRAPER")
    print("Snyk + Socket.dev + Checkmarx")
    print("=" * 60)

    # Load existing
    existing = []
    try:
        df = pd.read_csv(args.input)
        existing = df.to_dict('records')
        print(f"\nLoaded existing: {len(existing)} entries")
    except FileNotFoundError:
        print(f"Warning: {args.input} not found")

    # Scrape all sources
    new_entries = []
    new_entries.extend(scrape_snyk())
    new_entries.extend(scrape_socket())
    new_entries.extend(scrape_checkmarx())

    print(f"\nTotal new scraped: {len(new_entries)}")

    # Combine and deduplicate
    all_entries = existing + new_entries
    all_entries = deduplicate(all_entries)

    # Label new entries
    labeled, uncertain = label_entries(all_entries)

    # Stats
    dist = Counter(e["label"] for e in labeled
                   if e.get("label") in VALID_LABELS)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Total labeled: {len(labeled)}")
    print(f"Uncertain: {len(uncertain)}")
    print(f"\nLabel distribution:")
    for label in VALID_LABELS:
        count = dist.get(label, 0)
        pct = count/len(labeled)*100 if labeled else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Save
    cols = ["id", "title", "description", "label",
            "predicted_label", "source", "ecosystem", "year"]
    df_out = pd.DataFrame(labeled)
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out[cols].to_csv(args.output, index=False)
    print(f"\nSaved {len(labeled)} to {args.output}")

    if uncertain:
        rev = args.output.replace('.csv', '_review.csv')
        df_rev = pd.DataFrame(uncertain)
        for col in cols:
            if col not in df_rev.columns:
                df_rev[col] = ""
        df_rev[cols].to_csv(rev, index=False)
        print(f"Saved {len(uncertain)} uncertain to {rev}")

    target = 10000
    if len(labeled) >= target:
        print(f"\n✅ Target of {target} reached!")
    else:
        needed = target - len(labeled)
        print(f"\n⚠️ Got {len(labeled)} of {target}. Need {needed} more.")

    print("\nDone.")

if __name__ == "__main__":
    main()
