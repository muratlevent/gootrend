#!/usr/bin/env python3
"""
Google Trends LinkedIn Keywords Scraper
========================================
Scrapes trending keywords from Google Trends categories relevant to
LinkedIn content creation. Features interactive timeframe selection,
checkpoint/resume system, and rich terminal visualization.

Usage:
    python scraper.py              # Interactive mode
    python scraper.py --status     # Show checkpoint progress
    python scraper.py --phase 1    # Run specific phase
"""

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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.prompt import Prompt, Confirm
from rich.logging import RichHandler
from rich import box

# ---------------------------------------------------------------------------
# Console & Logging
# ---------------------------------------------------------------------------

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger("gootrend")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

TIMEFRAMES = {
    "now 7-d":   "Last 7 days",
    "today 1-m": "Last 1 month",
    "today 3-m": "Last 3 months",
}

GEOS = {
    "":   "Global",
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
}

CATEGORIES = {
    # Core business & LinkedIn
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
    # Technology Subcategories
    708:  "Software Development",
    1227: "Enterprise Technology",
    1299: "Artificial Intelligence",
    # Business Subcategories
    355:  "Logistics",
    96:   "Advertising",
    1159: "Startups",
    # Education Subcategories
    808:  "Certifications",
    260:  "Vocational & Continuing Education",
}

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
    # Tools & Platforms
    "Excel", "Power BI", "Tableau", "Python", "SQL",
    "Notion", "Jira", "Slack", "Zapier", "Canva",
    "Figma", "Adobe", "Shopify", "WordPress",
    "HubSpot", "Salesforce", "Google Analytics", "Semrush",
    "Midjourney", "Stable Diffusion", "Copilot", "Claude AI",
    # Roles & Titles
    "Product Manager", "Project Manager", "Data Analyst",
    "Software Engineer", "Digital Marketer", "HR Manager",
    "Scrum Master", "Business Analyst", "Content Creator",
    "UX Designer", "Sales Representative", "CEO", "Founder",
    # Intent Phrases
    "best tools for", "how to learn", "salary for",
    "certification", "course", "interview questions",
    "resume template", "remote jobs", "freelance sites",
]

# Rate limiting
MIN_DELAY = 3
MAX_DELAY = 7
RETRY_DELAY = 60
MAX_RETRIES = 3

# Save checkpoint every N combos
SAVE_EVERY = 50


# ---------------------------------------------------------------------------
# Checkpoint System  (uses sets internally for O(1) lookups)
# ---------------------------------------------------------------------------

_SECTION_KEYS = [
    "completed_related_queries",
    "completed_related_topics",
    "completed_trending_rss",
    "completed_category_trends",
]


class Checkpoint:
    """Tracks scraping progress with resume capability."""

    def __init__(self, path: Path = CHECKPOINT_FILE):
        self.path = path
        self.data = self._load()

    # -- persistence ---------------------------------------------------------

    def _empty(self) -> dict:
        return {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": None,
            "selected_timeframes": [],
            "completed_related_queries": set(),
            "completed_related_topics": set(),
            "completed_trending_rss": set(),
            "completed_category_trends": set(),
            "errors": [],
            "stats": {
                "total_related_queries_found": 0,
                "total_related_topics_found": 0,
                "total_trending_found": 0,
                "total_category_trends_found": 0,
            },
        }

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                raw = json.load(f)
            for k in _SECTION_KEYS:
                raw[k] = set(raw.get(k, []))
            return raw
        return self._empty()

    def save(self):
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = {}
        for k, v in self.data.items():
            out[k] = sorted(v) if isinstance(v, set) else v
        with open(self.path, "w") as f:
            json.dump(out, f, indent=2)

    # -- progress tracking ---------------------------------------------------

    def is_done(self, section: str, key: str) -> bool:
        return key in self.data.get(section, set())

    def mark_done(self, section: str, key: str):
        self.data[section].add(key)

    def log_error(self, context: str, error: str):
        self.data["errors"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "error": error[:200],
        })

    def incr_stat(self, stat: str, count: int):
        self.data["stats"][stat] = self.data["stats"].get(stat, 0) + count

    def reset(self):
        if self.path.exists():
            self.path.unlink()
        self.data = self._empty()

    def has_progress(self) -> bool:
        return self.path.exists() and any(
            len(self.data.get(s, set())) > 0 for s in _SECTION_KEYS
        )

    # -- display -------------------------------------------------------------

    def print_status(self, timeframes: dict):
        num_tf = max(len(timeframes), 1)
        combo_total = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES) * num_tf

        tbl = Table(
            title="📊 Checkpoint Status", box=box.ROUNDED,
            show_header=True, header_style="bold cyan",
        )
        tbl.add_column("Phase", style="bold")
        tbl.add_column("Done", justify="right")
        tbl.add_column("Total", justify="right")
        tbl.add_column("", justify="center")

        def icon(done, total):
            if done >= total:
                return "✅"
            return "🔄" if done > 0 else "⬜"

        rq = len(self.data.get("completed_related_queries", set()))
        rt = len(self.data.get("completed_related_topics", set()))
        rss = len(self.data.get("completed_trending_rss", set()))
        ct = len(self.data.get("completed_category_trends", set()))
        rss_total = len(GEOS) * 2

        tbl.add_row("Related Queries", f"{rq:,}", f"{combo_total:,}", icon(rq, combo_total))
        tbl.add_row("Related Topics", f"{rt:,}", f"{combo_total:,}", icon(rt, combo_total))
        tbl.add_row("Trending RSS", f"{rss:,}", f"{rss_total:,}", icon(rss, rss_total))
        tbl.add_row("Category Trends", f"{ct:,}", f"{len(GEOS)*len(CATEGORIES):,}", icon(ct, len(GEOS)*len(CATEGORIES)))
        console.print()
        console.print(tbl)

        stbl = Table(title="📈 Keywords Found", box=box.SIMPLE, header_style="bold green")
        stbl.add_column("Source")
        stbl.add_column("Count", justify="right")
        for k, v in self.data.get("stats", {}).items():
            label = k.replace("total_", "").replace("_found", "").replace("_", " ").title()
            stbl.add_row(label, f"{v:,}")
        console.print(stbl)

        errs = len(self.data.get("errors", []))
        if errs:
            console.print(f"\n[yellow]⚠  {errs} errors logged[/yellow]")
        console.print(f"\n[dim]Started: {self.data.get('started_at', 'N/A')}[/dim]")
        console.print(f"[dim]Last updated: {self.data.get('last_updated', 'N/A')}[/dim]\n")


# ---------------------------------------------------------------------------
# Interactive Menu
# ---------------------------------------------------------------------------

def show_banner():
    console.print()
    console.print(
        Panel(
            "[bold white]🔍  Google Trends LinkedIn Keywords Scraper[/bold white]\n"
            f"[dim]{len(SEED_KEYWORDS)} seeds  •  {len(CATEGORIES)} categories  •  {len(GEOS)} geos[/dim]",
            border_style="cyan",
            padding=(1, 3),
        )
    )


def select_timeframe() -> dict:
    """Let the user choose which timeframe(s) to scrape."""
    console.print("\n[bold cyan]Select timeframe:[/bold cyan]\n")
    opts = list(TIMEFRAMES.items())
    for i, (_, label) in enumerate(opts, 1):
        console.print(f"  [bold white][{i}][/bold white]  {label}")
    console.print(f"  [bold white][{len(opts)+1}][/bold white]  Full Report (all timeframes)")
    console.print()

    while True:
        raw = Prompt.ask("[bold yellow]▶[/bold yellow] Your choice", default=str(len(opts)+1))
        try:
            idx = int(raw)
        except ValueError:
            console.print("[red]  Please enter a number.[/red]")
            continue
        if 1 <= idx <= len(opts):
            code, label = opts[idx - 1]
            console.print(f"\n[green]✓ Selected:[/green] {label}\n")
            return {code: label}
        if idx == len(opts) + 1:
            console.print(f"\n[green]✓ Selected:[/green] Full Report (all timeframes)\n")
            return dict(TIMEFRAMES)
        console.print("[red]  Invalid choice.[/red]")


def handle_checkpoint_resume(checkpoint: Checkpoint) -> bool:
    """If progress exists, ask user to continue or start fresh.
    Returns True to continue, False to reset."""
    if not checkpoint.has_progress():
        return False

    summary = checkpoint.data
    total = sum(len(summary.get(s, set())) for s in _SECTION_KEYS)
    errs = len(summary.get("errors", []))

    console.print(
        Panel(
            f"[yellow]⚠  Previous session detected[/yellow]\n\n"
            f"  Started:    [dim]{summary.get('started_at','?')}[/dim]\n"
            f"  Updated:    [dim]{summary.get('last_updated','?')}[/dim]\n"
            f"  Completed:  [bold]{total:,}[/bold] combos\n"
            f"  Errors:     [dim]{errs}[/dim]",
            title="Session Recovery",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print("  [bold white][1][/bold white]  Continue from where you left off")
    console.print("  [bold white][2][/bold white]  Start fresh (delete all previous data)\n")

    while True:
        choice = Prompt.ask("[bold yellow]▶[/bold yellow] Your choice", choices=["1", "2"])
        if choice == "1":
            console.print("[green]✓ Resuming previous session…[/green]\n")
            return True
        sure = Confirm.ask("[red]  All previous data will be deleted. Are you sure?[/red]")
        if sure:
            return False
        console.print()


def show_config_summary(timeframes: dict):
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold cyan")
    tbl.add_column()
    tbl.add_row("Timeframes", ", ".join(timeframes.values()))
    tbl.add_row("Seeds", str(len(SEED_KEYWORDS)))
    tbl.add_row("Categories", str(len(CATEGORIES)))
    tbl.add_row("Geos", str(len(GEOS)))
    combo = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES) * len(timeframes)
    tbl.add_row("Phase 1/2 combos", f"{combo:,}")
    console.print(Panel(tbl, title="[bold]Configuration[/bold]", border_style="dim", padding=(0, 1)))
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_between_requests():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def safe_request(func, *args, retries=MAX_RETRIES, **kwargs):
    """Call a pytrends function with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            wait_between_requests()
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many" in err:
                wait = RETRY_DELAY * attempt
                log.warning(f"Rate limited (attempt {attempt}/{retries}). Waiting {wait}s…")
                time.sleep(wait)
            elif "response is empty" in err.lower() or "no data" in err.lower():
                return None
            else:
                log.warning(f"Error (attempt {attempt}/{retries}): {err[:120]}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY // 2)
                else:
                    raise
    return None


def create_pytrends():
    return TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1)


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=35),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _save_csv(records: list, path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Phase 1 – Related Queries
# ---------------------------------------------------------------------------

def scrape_related_queries(checkpoint: Checkpoint, timeframes: dict):
    console.rule("[bold cyan]Phase 1 — Related Queries[/bold cyan]")

    results = []
    csv_path = OUTPUT_DIR / "related_queries.csv"
    if csv_path.exists():
        results = pd.read_csv(csv_path).to_dict("records")

    pytrends = create_pytrends()
    total = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES) * len(timeframes)

    # Pre-count completed
    skip = sum(
        1 for k in checkpoint.data.get("completed_related_queries", set())
        if k.rsplit("|", 1)[-1] in timeframes
    )

    counter = 0
    with create_progress() as prog:
        task = prog.add_task("Related Queries", total=total, completed=skip)

        for tf, tf_label in timeframes.items():
            for kw in SEED_KEYWORDS:
                for geo_code, geo_name in GEOS.items():
                    for cat_id, cat_name in CATEGORIES.items():
                        combo = f"{kw}|{geo_code}|{cat_id}|{tf}"
                        if checkpoint.is_done("completed_related_queries", combo):
                            continue

                        try:
                            pytrends.build_payload([kw], cat=cat_id, timeframe=tf, geo=geo_code)
                            data = safe_request(pytrends.related_queries)
                            if data and kw in data:
                                for qtype in ("top", "rising"):
                                    df = data[kw].get(qtype)
                                    if df is not None and not df.empty:
                                        for _, row in df.iterrows():
                                            results.append({
                                                "seed_keyword": kw,
                                                "geo": geo_code or "Global",
                                                "geo_name": geo_name,
                                                "category_id": cat_id,
                                                "category_name": cat_name,
                                                "timeframe": tf,
                                                "timeframe_label": tf_label,
                                                "type": qtype,
                                                "query": row.get("query", ""),
                                                "value": row.get("value", ""),
                                            })
                                        checkpoint.incr_stat("total_related_queries_found", len(df))
                        except Exception as e:
                            checkpoint.log_error(f"rq|{combo}", str(e))

                        checkpoint.mark_done("completed_related_queries", combo)
                        prog.advance(task)
                        counter += 1
                        if counter % SAVE_EVERY == 0:
                            _save_csv(results, csv_path)
                            checkpoint.save()

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Related queries: {len(results):,} rows saved")
    return results


# ---------------------------------------------------------------------------
# Phase 2 – Related Topics
# ---------------------------------------------------------------------------

def scrape_related_topics(checkpoint: Checkpoint, timeframes: dict):
    console.rule("[bold cyan]Phase 2 — Related Topics[/bold cyan]")

    results = []
    csv_path = OUTPUT_DIR / "related_topics.csv"
    if csv_path.exists():
        results = pd.read_csv(csv_path).to_dict("records")

    pytrends = create_pytrends()
    total = len(SEED_KEYWORDS) * len(GEOS) * len(CATEGORIES) * len(timeframes)

    skip = sum(
        1 for k in checkpoint.data.get("completed_related_topics", set())
        if k.rsplit("|", 1)[-1] in timeframes
    )

    counter = 0
    with create_progress() as prog:
        task = prog.add_task("Related Topics", total=total, completed=skip)

        for tf, tf_label in timeframes.items():
            for kw in SEED_KEYWORDS:
                for geo_code, geo_name in GEOS.items():
                    for cat_id, cat_name in CATEGORIES.items():
                        combo = f"{kw}|{geo_code}|{cat_id}|{tf}"
                        if checkpoint.is_done("completed_related_topics", combo):
                            continue

                        try:
                            pytrends.build_payload([kw], cat=cat_id, timeframe=tf, geo=geo_code)
                            data = safe_request(pytrends.related_topics)
                            if data and kw in data:
                                for ttype in ("top", "rising"):
                                    df = data[kw].get(ttype)
                                    if df is not None and not df.empty:
                                        for _, row in df.iterrows():
                                            results.append({
                                                "seed_keyword": kw,
                                                "geo": geo_code or "Global",
                                                "geo_name": geo_name,
                                                "category_id": cat_id,
                                                "category_name": cat_name,
                                                "timeframe": tf,
                                                "timeframe_label": tf_label,
                                                "type": ttype,
                                                "topic_title": row.get("topic_title", ""),
                                                "topic_mid": row.get("topic_mid", ""),
                                                "topic_type": row.get("topic_type", ""),
                                                "value": row.get("value", ""),
                                            })
                                        checkpoint.incr_stat("total_related_topics_found", len(df))
                        except Exception as e:
                            checkpoint.log_error(f"rt|{combo}", str(e))

                        checkpoint.mark_done("completed_related_topics", combo)
                        prog.advance(task)
                        counter += 1
                        if counter % SAVE_EVERY == 0:
                            _save_csv(results, csv_path)
                            checkpoint.save()

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Related topics: {len(results):,} rows saved")
    return results


# ---------------------------------------------------------------------------
# Phase 3 – Trending RSS  (real-time, no timeframe dimension)
# ---------------------------------------------------------------------------

def scrape_trending_rss(checkpoint: Checkpoint, _timeframes: dict):
    console.rule("[bold cyan]Phase 3 — Real-Time Trending Searches[/bold cyan]")

    results = []
    csv_path = OUTPUT_DIR / "trending_searches.csv"
    if csv_path.exists():
        results = pd.read_csv(csv_path).to_dict("records")

    pytrends = create_pytrends()
    total = len(GEOS) * 2  # daily + realtime per geo

    skip = len(checkpoint.data.get("completed_trending_rss", set()))
    with create_progress() as prog:
        task = prog.add_task("Trending RSS", total=total, completed=skip)

        # -- daily trending --
        for geo_code, geo_name in GEOS.items():
            if checkpoint.is_done("completed_trending_rss", geo_code):
                continue
            pn = geo_code or "US"
            try:
                df = safe_request(pytrends.trending_searches, pn=pn)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        results.append({
                            "geo": geo_code or "Global",
                            "geo_name": geo_name,
                            "trending_keyword": row.iloc[0] if len(row) > 0 else "",
                        })
                    checkpoint.incr_stat("total_trending_found", len(df))
            except Exception as e:
                checkpoint.log_error(f"trending|{geo_code}", str(e))
            checkpoint.mark_done("completed_trending_rss", geo_code)
            prog.advance(task)

        # -- realtime trending --
        for geo_code, geo_name in GEOS.items():
            rss_key = f"realtime_{geo_code}"
            if checkpoint.is_done("completed_trending_rss", rss_key):
                continue
            pn = geo_code or "US"
            try:
                df = safe_request(pytrends.realtime_trending_searches, pn=pn)
                if df is not None and not df.empty:
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
            except Exception as e:
                checkpoint.log_error(f"realtime|{geo_code}", str(e))
            checkpoint.mark_done("completed_trending_rss", rss_key)
            prog.advance(task)

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Trending searches: {len(results):,} rows saved")
    return results


# ---------------------------------------------------------------------------
# Phase 4 – Category Trends  (suggestions API is timeframe-independent)
# ---------------------------------------------------------------------------

def scrape_category_trends(checkpoint: Checkpoint, timeframes: dict):
    console.rule("[bold cyan]Phase 4 — Category-Based Trends[/bold cyan]")

    results = []
    csv_path = OUTPUT_DIR / "category_trends.csv"
    if csv_path.exists():
        results = pd.read_csv(csv_path).to_dict("records")

    pytrends = create_pytrends()
    default_tf = next(iter(timeframes))

    category_anchors = [
        "business", "technology", "career", "marketing",
        "leadership", "finance", "education", "software",
    ]

    total = len(GEOS) * len(CATEGORIES)
    skip = len(checkpoint.data.get("completed_category_trends", set()))
    counter = 0

    with create_progress() as prog:
        task = prog.add_task("Category Trends", total=total, completed=skip)

        for geo_code, geo_name in GEOS.items():
            for cat_id, cat_name in CATEGORIES.items():
                combo = f"{geo_code}|{cat_id}"
                if checkpoint.is_done("completed_category_trends", combo):
                    continue

                try:
                    pytrends.build_payload(
                        category_anchors[:5], cat=cat_id,
                        timeframe=default_tf, geo=geo_code,
                    )
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
                                    checkpoint.incr_stat("total_category_trends_found", 1)
                        except Exception:
                            pass
                        wait_between_requests()
                except Exception as e:
                    checkpoint.log_error(f"cat|{combo}", str(e))

                checkpoint.mark_done("completed_category_trends", combo)
                prog.advance(task)
                counter += 1
                if counter % SAVE_EVERY == 0:
                    _save_csv(results, csv_path)
                    checkpoint.save()

    _save_csv(results, csv_path)
    checkpoint.save()
    log.info(f"Category trends: {len(results):,} rows saved")
    return results


# ---------------------------------------------------------------------------
# Phase 5 – Master Keywords Aggregation
# ---------------------------------------------------------------------------

def build_master_keywords(checkpoint: Checkpoint):
    console.rule("[bold cyan]Phase 5 — Building Master Keywords List[/bold cyan]")
    all_kw: set[str] = set()

    sources = [
        ("related_queries.csv",  "query"),
        ("related_topics.csv",   "topic_title"),
        ("trending_searches.csv","trending_keyword"),
        ("category_trends.csv",  "suggestion_title"),
    ]

    for fname, col in sources:
        p = OUTPUT_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            if col in df.columns:
                all_kw.update(df[col].dropna().str.strip().str.lower().unique())
            log.info(f"  {fname}: {len(df):,} rows")

    for kw in SEED_KEYWORDS:
        all_kw.add(kw.lower().strip())
    all_kw.discard("")

    sorted_kw = sorted(all_kw)
    master_path = OUTPUT_DIR / "all_keywords_master.csv"
    pd.DataFrame({"keyword": sorted_kw}).to_csv(master_path, index=False, encoding="utf-8-sig")

    console.print(
        Panel(
            f"[bold green]{len(sorted_kw):,}[/bold green] unique keywords\n"
            f"[dim]Saved to {master_path}[/dim]",
            title="Master List",
            border_style="green",
        )
    )
    return sorted_kw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run():
    parser = argparse.ArgumentParser(description="Google Trends LinkedIn Keywords Scraper")
    parser.add_argument("--status", action="store_true", help="Show checkpoint progress")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific phase")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint()

    # -- status only ---------------------------------------------------------
    if args.status:
        show_banner()
        stored = checkpoint.data.get("selected_timeframes", [])
        tf = {k: TIMEFRAMES[k] for k in stored if k in TIMEFRAMES} or dict(TIMEFRAMES)
        checkpoint.print_status(tf)
        return

    # -- interactive start ---------------------------------------------------
    show_banner()

    # Resume or reset?
    if checkpoint.has_progress():
        should_continue = handle_checkpoint_resume(checkpoint)
        if not should_continue:
            checkpoint.reset()
            for f in OUTPUT_DIR.glob("*.csv"):
                f.unlink()
            console.print("[dim]Previous data cleared.[/dim]\n")
            checkpoint = Checkpoint()
    else:
        console.print("[dim]No previous session found — starting fresh.[/dim]\n")

    # Determine timeframes
    stored_tf = checkpoint.data.get("selected_timeframes", [])
    if stored_tf and checkpoint.has_progress():
        timeframes = {k: TIMEFRAMES[k] for k in stored_tf if k in TIMEFRAMES}
        console.print(f"[dim]Resuming with timeframes: {', '.join(timeframes.values())}[/dim]\n")
    else:
        timeframes = select_timeframe()
        checkpoint.data["selected_timeframes"] = list(timeframes.keys())
        checkpoint.save()

    show_config_summary(timeframes)

    # -- run phases ----------------------------------------------------------
    phases = {
        1: ("Related Queries", scrape_related_queries),
        2: ("Related Topics",  scrape_related_topics),
        3: ("Trending RSS",    scrape_trending_rss),
        4: ("Category Trends", scrape_category_trends),
    }

    if args.phase:
        if args.phase == 5:
            build_master_keywords(checkpoint)
        elif args.phase in phases:
            name, func = phases[args.phase]
            func(checkpoint, timeframes)
            build_master_keywords(checkpoint)
    else:
        for num, (name, func) in phases.items():
            func(checkpoint, timeframes)
        build_master_keywords(checkpoint)

    # -- final summary -------------------------------------------------------
    checkpoint.print_status(timeframes)
    console.print("[bold green]✅  Done! Check the output/ directory for results.[/bold green]\n")


def main():
    try:
        _run()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠  Interrupted! Progress saved to checkpoint.[/yellow]")
        console.print("[dim]Run again to resume from where you left off.[/dim]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
