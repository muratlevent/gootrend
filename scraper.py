#!/usr/bin/env python3
"""
Google Trends LinkedIn Keywords Scraper
========================================
Scrapes trending keywords from Google Trends categories relevant to
LinkedIn content creation. Covers global + English-speaking geos,
last 3 months. Includes checkpoint/resume system.

Usage:
    python scraper.py              # Full scrape (resumes from checkpoint)
    python scraper.py --reset      # Clear checkpoint and start fresh
    python scraper.py --status     # Show checkpoint progress
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

TIMEFRAME = "today 3-m"  # last 3 months

# Geographic scopes
GEOS = {
    "": "Global",
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
}

# Google Trends category IDs — expanded set for maximum keyword coverage
CATEGORIES = {
    # Core business & LinkedIn categories
    12:   "Business & Industrial",
    88:   "Business Services",
    152:  "Human Resources",
    1199: "Management",
    1279: "Small Business",
    1200: "Business Operations",
    784:  "Professional & Trade Associations",
    903:  "Corporate Training",
    25:   "Advertising & Marketing",
    # Jobs & Education
    22:   "Jobs & Education",
    958:  "Jobs & Education (sub)",
    # Tech & Internet
    5:    "Computers & Electronics",
    13:   "Internet & Telecom",
    32:   "Software",
    # Finance, Law, Science, Shopping
    7:    "Finance",
    19:   "Law & Government",
    174:  "Science",
    18:   "Shopping",
    # Community & Reference
    299:  "Online Communities",
    533:  "Reference",
}

# Seed keywords — comprehensive set organized by theme
SEED_KEYWORDS = [
    # LinkedIn Core
    "linkedin", "linkedin strategy", "linkedin growth", "linkedin algorithm",
    "linkedin content", "linkedin engagement", "linkedin profile optimization",
    "linkedin networking", "linkedin tips", "linkedin marketing",
    "personal branding", "thought leadership", "linkedin creator",
    "linkedin post", "linkedin analytics",
    # Career & Jobs
    "hiring trends", "job market", "career growth", "career change",
    "resume tips", "job interview tips", "remote work", "hybrid work",
    "salary negotiation", "layoffs", "talent acquisition",
    "employee retention", "skills gap", "upskilling", "reskilling",
    "job search", "freelancing", "gig economy", "career development",
    "work from home",
    # Business & Leadership
    "leadership", "management", "business strategy", "entrepreneurship",
    "startup", "small business", "innovation", "digital transformation",
    "business growth", "scaling business", "company culture",
    "team building", "change management", "corporate strategy",
    "business development", "project management", "agile methodology",
    "organizational design", "strategic planning",
    # Marketing & Sales
    "digital marketing", "content marketing", "SEO", "social media marketing",
    "email marketing", "brand strategy", "influencer marketing",
    "growth hacking", "conversion optimization", "sales strategy",
    "B2B marketing", "B2B sales", "account based marketing",
    "marketing automation", "copywriting", "storytelling in business",
    # Technology & AI
    "artificial intelligence", "AI in business", "machine learning",
    "generative AI", "ChatGPT", "SaaS", "automation", "data analytics",
    "big data", "cloud computing", "cybersecurity", "blockchain",
    "no code", "low code", "productivity tools", "AI tools",
    "tech industry", "software development", "web development",
    "product management",
    # Economy & Finance
    "economy", "recession", "inflation", "venture capital", "investment",
    "stock market", "cryptocurrency", "fintech", "financial planning",
    "business finance", "funding", "IPO",
    # Workplace & Culture
    "sustainability", "ESG", "diversity inclusion", "work life balance",
    "mental health at work", "future of work", "employee engagement",
    "workplace culture", "burnout", "remote team management",
    "employee experience", "workforce planning",
    # Industry Topics
    "ecommerce", "retail trends", "supply chain", "logistics",
    "healthcare industry", "real estate", "consulting", "legal tech",
    "education technology", "edtech", "online learning",
    "professional development", "networking events", "conferences",
    # Emerging Trends
    "creator economy", "side hustle", "passive income",
    "personal finance", "coaching business", "online business",
    "digital nomad", "solopreneur", "bootstrapping",
]

# Rate limiting
MIN_DELAY = 3   # minimum seconds between requests
MAX_DELAY = 7   # maximum seconds between requests
RETRY_DELAY = 60  # seconds to wait on 429 error
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint System
# ---------------------------------------------------------------------------

class Checkpoint:
    """Tracks scraping progress for resume capability."""

    def __init__(self, path: Path = CHECKPOINT_FILE):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": None,
            "completed_related_queries": [],   # list of "keyword|geo|cat"
            "completed_related_topics": [],    # list of "keyword|geo|cat"
            "completed_trending_rss": [],      # list of geo codes
            "completed_category_trends": [],   # list of "geo|cat"
            "errors": [],
            "stats": {
                "total_related_queries_found": 0,
                "total_related_topics_found": 0,
                "total_trending_found": 0,
                "total_category_trends_found": 0,
            },
        }

    def save(self):
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_done(self, section: str, key: str) -> bool:
        return key in self.data.get(section, [])

    def mark_done(self, section: str, key: str):
        if key not in self.data[section]:
            self.data[section].append(key)

    def log_error(self, context: str, error: str):
        self.data["errors"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "error": error,
        })

    def incr_stat(self, stat: str, count: int):
        self.data["stats"][stat] = self.data["stats"].get(stat, 0) + count

    def reset(self):
        if self.path.exists():
            self.path.unlink()
        self.data = self._load()

    def print_status(self):
        rq = len(self.data.get("completed_related_queries", []))
        rt = len(self.data.get("completed_related_topics", []))
        rss = len(self.data.get("completed_trending_rss", []))
        ct = len(self.data.get("completed_category_trends", []))
        total_combos = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES)
        print(f"\n{'='*60}")
        print(f"  CHECKPOINT STATUS")
        print(f"{'='*60}")
        print(f"  Started:           {self.data.get('started_at', 'N/A')}")
        print(f"  Last updated:      {self.data.get('last_updated', 'N/A')}")
        print(f"  Related queries:   {rq}/{total_combos} combos done")
        print(f"  Related topics:    {rt}/{total_combos} combos done")
        print(f"  Trending RSS:      {rss}/{len(GEOS)} geos done")
        print(f"  Category trends:   {ct}/{len(GEOS)*len(CATEGORIES)} combos done")
        print(f"  Errors logged:     {len(self.data.get('errors', []))}")
        print(f"\n  Keywords found so far:")
        for k, v in self.data.get("stats", {}).items():
            print(f"    {k}: {v}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Helper: rate-limited request with retries
# ---------------------------------------------------------------------------

def wait_between_requests():
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)


def safe_request(func, *args, retries=MAX_RETRIES, **kwargs):
    """Call a pytrends function with retry logic on 429/connection errors."""
    for attempt in range(1, retries + 1):
        try:
            wait_between_requests()
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many" in err_str:
                wait = RETRY_DELAY * attempt
                log.warning(f"Rate limited (attempt {attempt}/{retries}). "
                            f"Waiting {wait}s...")
                time.sleep(wait)
            elif "response is empty" in err_str.lower() or "no data" in err_str.lower():
                log.info(f"No data returned: {err_str}")
                return None
            else:
                log.warning(f"Error (attempt {attempt}/{retries}): {err_str}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY // 2)
                else:
                    raise
    return None


# ---------------------------------------------------------------------------
# Scraping Functions
# ---------------------------------------------------------------------------

def create_pytrends():
    """Create a new TrendReq session."""
    return TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1)


def scrape_related_queries(checkpoint: Checkpoint):
    """Scrape related queries for every (keyword, geo, category) combo."""
    log.info("=" * 60)
    log.info("PHASE 1: Scraping Related Queries")
    log.info("=" * 60)

    results = []
    csv_path = OUTPUT_DIR / "related_queries.csv"

    # Load existing results if resuming
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        results = existing.to_dict("records")
        log.info(f"Loaded {len(results)} existing related query rows")

    pytrends = create_pytrends()
    total = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES)
    done_count = len(checkpoint.data.get("completed_related_queries", []))

    for kw in SEED_KEYWORDS:
        for geo_code, geo_name in GEOS.items():
            for cat_id, cat_name in CATEGORIES.items():
                combo_key = f"{kw}|{geo_code}|{cat_id}"
                if checkpoint.is_done("completed_related_queries", combo_key):
                    continue

                done_count += 1
                log.info(f"[RQ {done_count}/{total}] '{kw}' | {geo_name} | {cat_name}")

                try:
                    pytrends.build_payload(
                        [kw],
                        cat=cat_id,
                        timeframe=TIMEFRAME,
                        geo=geo_code,
                    )
                    data = safe_request(pytrends.related_queries)
                    if data and kw in data:
                        for query_type in ["top", "rising"]:
                            df = data[kw].get(query_type)
                            if df is not None and not df.empty:
                                count = len(df)
                                for _, row in df.iterrows():
                                    results.append({
                                        "seed_keyword": kw,
                                        "geo": geo_code or "Global",
                                        "geo_name": geo_name,
                                        "category_id": cat_id,
                                        "category_name": cat_name,
                                        "type": query_type,
                                        "query": row.get("query", ""),
                                        "value": row.get("value", ""),
                                    })
                                checkpoint.incr_stat(
                                    "total_related_queries_found", count
                                )
                                log.info(f"  -> Found {count} {query_type} queries")
                except Exception as e:
                    log.error(f"  -> Error: {e}")
                    checkpoint.log_error(f"related_queries|{combo_key}", str(e))

                checkpoint.mark_done("completed_related_queries", combo_key)

                # Save periodically (every combo)
                if done_count % 10 == 0:
                    _save_csv(results, csv_path)
                    checkpoint.save()

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Related queries: {len(results)} total rows saved")
    return results


def scrape_related_topics(checkpoint: Checkpoint):
    """Scrape related topics for every (keyword, geo, category) combo."""
    log.info("=" * 60)
    log.info("PHASE 2: Scraping Related Topics")
    log.info("=" * 60)

    results = []
    csv_path = OUTPUT_DIR / "related_topics.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        results = existing.to_dict("records")
        log.info(f"Loaded {len(results)} existing related topic rows")

    pytrends = create_pytrends()
    total = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES)
    done_count = len(checkpoint.data.get("completed_related_topics", []))

    for kw in SEED_KEYWORDS:
        for geo_code, geo_name in GEOS.items():
            for cat_id, cat_name in CATEGORIES.items():
                combo_key = f"{kw}|{geo_code}|{cat_id}"
                if checkpoint.is_done("completed_related_topics", combo_key):
                    continue

                done_count += 1
                log.info(f"[RT {done_count}/{total}] '{kw}' | {geo_name} | {cat_name}")

                try:
                    pytrends.build_payload(
                        [kw],
                        cat=cat_id,
                        timeframe=TIMEFRAME,
                        geo=geo_code,
                    )
                    data = safe_request(pytrends.related_topics)
                    if data and kw in data:
                        for topic_type in ["top", "rising"]:
                            df = data[kw].get(topic_type)
                            if df is not None and not df.empty:
                                count = len(df)
                                for _, row in df.iterrows():
                                    results.append({
                                        "seed_keyword": kw,
                                        "geo": geo_code or "Global",
                                        "geo_name": geo_name,
                                        "category_id": cat_id,
                                        "category_name": cat_name,
                                        "type": topic_type,
                                        "topic_title": row.get("topic_title", ""),
                                        "topic_mid": row.get("topic_mid", ""),
                                        "topic_type": row.get("topic_type", ""),
                                        "value": row.get("value", ""),
                                    })
                                checkpoint.incr_stat(
                                    "total_related_topics_found", count
                                )
                                log.info(f"  -> Found {count} {topic_type} topics")
                except Exception as e:
                    log.error(f"  -> Error: {e}")
                    checkpoint.log_error(f"related_topics|{combo_key}", str(e))

                checkpoint.mark_done("completed_related_topics", combo_key)

                if done_count % 10 == 0:
                    _save_csv(results, csv_path)
                    checkpoint.save()

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Related topics: {len(results)} total rows saved")
    return results


def scrape_trending_rss(checkpoint: Checkpoint):
    """Scrape real-time trending searches via RSS for each geo."""
    log.info("=" * 60)
    log.info("PHASE 3: Scraping Real-Time Trending Searches (RSS)")
    log.info("=" * 60)

    results = []
    csv_path = OUTPUT_DIR / "trending_searches.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        results = existing.to_dict("records")

    pytrends = create_pytrends()

    for geo_code, geo_name in GEOS.items():
        if checkpoint.is_done("completed_trending_rss", geo_code):
            continue

        # RSS trending only works for specific countries (not global "")
        pn = geo_code if geo_code else "US"
        log.info(f"[RSS] Fetching trending for {geo_name} (pn={pn})")

        try:
            # trending_searches returns a DataFrame
            df = safe_request(pytrends.trending_searches, pn=pn)
            if df is not None and not df.empty:
                count = len(df)
                for _, row in df.iterrows():
                    results.append({
                        "geo": geo_code or "Global",
                        "geo_name": geo_name,
                        "trending_keyword": row.iloc[0] if len(row) > 0 else "",
                    })
                checkpoint.incr_stat("total_trending_found", count)
                log.info(f"  -> Found {count} trending searches")
        except Exception as e:
            log.error(f"  -> Error: {e}")
            checkpoint.log_error(f"trending_rss|{geo_code}", str(e))

        checkpoint.mark_done("completed_trending_rss", geo_code)

    # Also try realtime trending searches for richer data
    for geo_code, geo_name in GEOS.items():
        rss_key = f"realtime_{geo_code}"
        if checkpoint.is_done("completed_trending_rss", rss_key):
            continue

        pn = geo_code if geo_code else "US"
        log.info(f"[RSS-Realtime] Fetching for {geo_name}")
        try:
            df = safe_request(
                pytrends.realtime_trending_searches, pn=pn
            )
            if df is not None and not df.empty:
                # realtime returns more columns
                for _, row in df.iterrows():
                    title = row.get("title", row.get("entityNames", ""))
                    if isinstance(title, list):
                        title = ", ".join(title)
                    results.append({
                        "geo": geo_code or "Global",
                        "geo_name": geo_name,
                        "trending_keyword": str(title),
                    })
                checkpoint.incr_stat("total_trending_found", len(df))
                log.info(f"  -> Found {len(df)} realtime trending")
        except Exception as e:
            log.info(f"  -> Realtime trending not available for {pn}: {e}")
            checkpoint.log_error(f"realtime_trending|{geo_code}", str(e))

        checkpoint.mark_done("completed_trending_rss", rss_key)

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Trending searches: {len(results)} total rows saved")
    return results


def scrape_category_trends(checkpoint: Checkpoint):
    """Scrape interest-over-time for each category (no keyword filter)
    to discover what's trending within each category."""
    log.info("=" * 60)
    log.info("PHASE 4: Scraping Category-Based Trends")
    log.info("=" * 60)

    results = []
    csv_path = OUTPUT_DIR / "category_trends.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        results = existing.to_dict("records")

    pytrends = create_pytrends()

    # For category trends, we use broad anchor keywords to discover
    # what's trending in each category
    category_anchors = [
        "business", "technology", "career", "marketing",
        "leadership", "finance", "education", "software",
    ]

    for geo_code, geo_name in GEOS.items():
        for cat_id, cat_name in CATEGORIES.items():
            combo_key = f"{geo_code}|{cat_id}"
            if checkpoint.is_done("completed_category_trends", combo_key):
                continue

            log.info(f"[CAT] {cat_name} | {geo_name}")

            try:
                # Use a broad keyword to explore the category
                pytrends.build_payload(
                    category_anchors[:5],  # max 5 keywords per payload
                    cat=cat_id,
                    timeframe=TIMEFRAME,
                    geo=geo_code,
                )

                # Get suggestions/related queries for this category
                for anchor in category_anchors[:5]:
                    try:
                        suggestions = pytrends.suggestions(anchor)
                        if suggestions:
                            for s in suggestions:
                                results.append({
                                    "geo": geo_code or "Global",
                                    "geo_name": geo_name,
                                    "category_id": cat_id,
                                    "category_name": cat_name,
                                    "anchor": anchor,
                                    "suggestion_title": s.get("title", ""),
                                    "suggestion_type": s.get("type", ""),
                                    "suggestion_mid": s.get("mid", ""),
                                })
                                checkpoint.incr_stat(
                                    "total_category_trends_found", 1
                                )
                    except Exception:
                        pass
                    wait_between_requests()

            except Exception as e:
                log.error(f"  -> Error: {e}")
                checkpoint.log_error(f"category_trends|{combo_key}", str(e))

            checkpoint.mark_done("completed_category_trends", combo_key)

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Category trends: {len(results)} total rows saved")
    return results


# ---------------------------------------------------------------------------
# Output & Aggregation
# ---------------------------------------------------------------------------

def _save_csv(records: list, path: Path):
    """Save a list of dicts to CSV."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def build_master_keywords(checkpoint: Checkpoint):
    """Combine all outputs into a single deduplicated master keyword list."""
    log.info("=" * 60)
    log.info("PHASE 5: Building Master Keywords List")
    log.info("=" * 60)

    all_keywords = set()

    # Related queries
    rq_path = OUTPUT_DIR / "related_queries.csv"
    if rq_path.exists():
        df = pd.read_csv(rq_path)
        if "query" in df.columns:
            all_keywords.update(
                df["query"].dropna().str.strip().str.lower().unique()
            )
        log.info(f"  Related queries contributed {len(df)} rows")

    # Related topics
    rt_path = OUTPUT_DIR / "related_topics.csv"
    if rt_path.exists():
        df = pd.read_csv(rt_path)
        if "topic_title" in df.columns:
            all_keywords.update(
                df["topic_title"].dropna().str.strip().str.lower().unique()
            )
        log.info(f"  Related topics contributed {len(df)} rows")

    # Trending searches
    ts_path = OUTPUT_DIR / "trending_searches.csv"
    if ts_path.exists():
        df = pd.read_csv(ts_path)
        if "trending_keyword" in df.columns:
            all_keywords.update(
                df["trending_keyword"].dropna().str.strip().str.lower().unique()
            )
        log.info(f"  Trending searches contributed {len(df)} rows")

    # Category trends
    ct_path = OUTPUT_DIR / "category_trends.csv"
    if ct_path.exists():
        df = pd.read_csv(ct_path)
        if "suggestion_title" in df.columns:
            all_keywords.update(
                df["suggestion_title"].dropna().str.strip().str.lower().unique()
            )
        log.info(f"  Category trends contributed {len(df)} rows")

    # Also include original seed keywords
    for kw in SEED_KEYWORDS:
        all_keywords.add(kw.lower().strip())

    # Remove empties
    all_keywords.discard("")

    # Sort and save
    sorted_keywords = sorted(all_keywords)
    master_df = pd.DataFrame({"keyword": sorted_keywords})
    master_path = OUTPUT_DIR / "all_keywords_master.csv"
    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")

    log.info(f"\n{'='*60}")
    log.info(f"  MASTER LIST: {len(sorted_keywords)} unique keywords")
    log.info(f"  Saved to: {master_path}")
    log.info(f"{'='*60}\n")

    return sorted_keywords


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Google Trends LinkedIn Keywords Scraper"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear checkpoint and start fresh"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show checkpoint progress and exit"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only a specific phase (1=queries, 2=topics, 3=rss, 4=category, 5=master)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint()

    if args.reset:
        checkpoint.reset()
        log.info("Checkpoint cleared. Starting fresh.")
        # Also remove old outputs
        for f in OUTPUT_DIR.glob("*.csv"):
            f.unlink()
        log.info("Old CSV files removed.")
        checkpoint = Checkpoint()

    if args.status:
        checkpoint.print_status()
        return

    log.info(f"Google Trends LinkedIn Scraper")
    log.info(f"Categories: {len(CATEGORIES)}")
    log.info(f"Seed keywords: {len(SEED_KEYWORDS)}")
    log.info(f"Geos: {len(GEOS)}")
    log.info(f"Timeframe: {TIMEFRAME}")
    log.info(f"Total combos (per phase): {len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES)}")
    log.info("")

    phases = {
        1: ("Related Queries", scrape_related_queries),
        2: ("Related Topics", scrape_related_topics),
        3: ("Trending RSS", scrape_trending_rss),
        4: ("Category Trends", scrape_category_trends),
    }

    if args.phase:
        if args.phase == 5:
            build_master_keywords(checkpoint)
        elif args.phase in phases:
            name, func = phases[args.phase]
            log.info(f"Running phase {args.phase}: {name}")
            func(checkpoint)
            build_master_keywords(checkpoint)
    else:
        # Run all phases
        for phase_num, (name, func) in phases.items():
            log.info(f"\n>>> Starting Phase {phase_num}: {name}\n")
            func(checkpoint)

        build_master_keywords(checkpoint)

    checkpoint.print_status()
    log.info("Done! Check the output/ directory for results.")


if __name__ == "__main__":
    main()
