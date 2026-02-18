# 🔍 Gootrend: Google Trends Keywords Scraper

A tool designed to extract trending keywords from Google Trends categories specifically relevant for content creators, marketers, and business professionals. This tool is built on top of the [pytrends](https://github.com/GeneralMills/pytrends) library, an unofficial API for Google Trends.

---

## 🌟 Features

-   **Interactive Selection**: Choose between multiple timeframes (7 days, 1 month, 3 months, or All).
-   **Phase-Based Scraping**:
    -   **Phase 1**: Related queries for over 180+ seed keywords.
    -   **Phase 2**: Category-based discovery using anchor topics.
    -   **Phase 3**: Automatic master list aggregation.
-   **Smart Checkpointing**: Never lose progress. If the script stops, you can resume exactly where you left off.
-   **Rate Limit Protection**: Built-in randomized delays and session refreshing to minimize blocks.
-   **Rich Visualization**: Professional terminal interface with progress bars and status reports using `rich`.
-   **CSV Outputs**: All data is exported to organized CSV files in the `output/` directory.

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gootrend.git
   cd gootrend
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠 Usage

### 1. Start Scraping (Interactive)
The most common way to run the tool. It will guide you through timeframe selection and session recovery.
```bash
python scraper.py
```

### 2. Check Progress
If you want to see how much of the work is completed without starting a new session:
```bash
python scraper.py --status
```

### 3. Specific Phases
You can run only a specific phase if needed (e.g., if you only want to rebuild the master list):
```bash
python scraper.py --phase 3
```

---

## 🛑 Rate Limiting & Unblocking Strategies

Google Trends (via the `pytrends` library) is extremely sensitive to automated requests. If you start seeing "Too Many Requests" (HTTP 429) or empty responses, follow these steps:

### 1. The Modem Trick (Highly Recommended)
If you have a dynamic IP address (common for residential internet), this is the most effective method:
- **Turn off your modem** (unplug the power).
- **Wait for at least 10-15 seconds.**
- **Turn it back on.**
- Your ISP will likely assign you a **new IP address**, which completely resets Google's rate limit for your connection.

### 2. VPN or Proxy
If you cannot reset your modem:
- Connect to a VPN and change your location.
- The script uses standard HTTP requests; if you set up a system-wide proxy, it will use that.
- *Tip*: Avoid common public data center proxies as Google often blocks them. Residential proxies are more effective.

### 3. Built-in Safeguards
The project already includes several features to help you avoid being banned:
- **Randomized Delays**: Wait times between 10-20 seconds between requests.
- **Session Refresh**: Every 100 requests, the script refreshes its cookies to avoid being flagged as a "stale" bot.
- **Retry Logic**: Automatically waits and retries on transient errors.

---

## 📁 Output Structure

All results are stored in the `output/` folder:
- `related_queries.csv`: Raw data for combined seed keywords and categories.
- `category_trends.csv`: Trends discovered through category anchors.
- `all_keywords_master.csv`: A clean, sorted, unique list of every keyword found during the process.
- `checkpoint.json`: Technical file used to track progress.

---

## ⚠️ Important Considerations

-   **Don't Rush**: Do not lower the `MIN_DELAY` or `MAX_DELAY` in the script. Speeding it up significantly increases the chance of a permanent per-session ban.
-   **Keywords**: The seed keywords are pre-configured (AI, SaaS, Career, Leadership). You can modify them directly in `scraper.py`.
-   **Maintenance**: Since this uses an unofficial API (`pytrends`), it may occasionally break if Google changes its internal Trend endpoints.

---

## 📜 License
This project is for educational and research purposes. Please respect Google's Terms of Service when using web scrapers.
