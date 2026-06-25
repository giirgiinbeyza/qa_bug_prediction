"""
data/github_issues.py
─────────────────────
GitHub Issues API'sinden "bug" etiketli issue'ları çeker,
NLP ile sentiment skoru hesaplar ve Promise dataset ile birleştirir.

GITHUB_TOKEN env değişkeni tanımlıysa → Gerçek API
Tanımlı değilse              → Simülasyon (aynı JSON yapısı)
"""

import os
import time
import requests
import numpy  as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    MODULES, NEGATIVE_KEYWORDS, NLP_SATURATION,
    GITHUB_REPO_OWNER, GITHUB_REPO_NAME,
    GITHUB_ISSUE_LABEL, GITHUB_MAX_PAGES, GITHUB_SIM_COUNT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GitHub API Çağrısı
# ─────────────────────────────────────────────────────────────────────────────

def fetch_github_issues_real(owner: str, repo: str,
                              label: str, max_pages: int) -> list[dict]:
    """
    GitHub REST API v3 — kapalı 'bug' issue'larını sayfalı olarak çeker.
    Rate limit: token olmadan 60 req/saat, token ile 5 000 req/saat.
    """
    token   = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    issues = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/issues"
            f"?labels={label}&state=closed&per_page=100&page={page}"
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if not data:
                    break
                issues.extend(data)
            else:
                print(f"   ⚠️  GitHub API {r.status_code}: {r.json().get('message','')}")
                break
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Bağlantı hatası: {e}")
            break
        time.sleep(0.5)   # Kibarca bekle — rate limit'e girme

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simülasyon (Token yokken)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_github_issues(n: int, modules: list) -> list[dict]:
    """
    GitHub API'nin döndürdüğü JSON yapısını birebir taklit eder.
    Gerçek token olmadan da pipeline'ın tamamı çalışsın diye vardır.
    """
    np.random.seed(99)

    templates = [
        "NullPointerException in {m} after update",
        "Critical: {m} crashes on concurrent requests",
        "{m} returns wrong data intermittently",
        "Memory leak detected in {m}",
        "Regression: {m} broken after v2.3 merge",
        "Security: {m} exposes sensitive endpoint",
        "{m} timeout under high load",
        "Test failure: {m} unit tests unstable",
        "Performance degradation in {m}",
        "Data corruption bug in {m}",
    ]
    labels_pool = [
        [{"name": "bug"}],
        [{"name": "bug"}, {"name": "critical"}],
        [{"name": "bug"}, {"name": "regression"}],
        [{"name": "bug"}, {"name": "security"}],
        [{"name": "bug"}, {"name": "performance"}],
    ]

    base_date = datetime(2024, 1, 1)
    result    = []

    for i in range(n):
        mod  = np.random.choice(modules)
        tmpl = np.random.choice(templates)
        days = np.random.randint(0, 365)
        severity = "critical" if np.random.rand() < 0.2 else "normal"

        result.append({
            "id"        : 1000 + i,
            "number"    : 1000 + i,
            "title"     : tmpl.format(m=mod),
            "body"      : (f"Steps to reproduce: [automatic] "
                           f"Affects module: {mod}. Severity: {severity}."),
            "labels"    : labels_pool[np.random.randint(len(labels_pool))],
            "created_at": (base_date + timedelta(days=days)).isoformat() + "Z",
            "closed_at" : (base_date + timedelta(
                            days=days + np.random.randint(1, 30))).isoformat() + "Z",
            "state"     : "closed",
            "comments"  : int(np.random.exponential(3)),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. NLP — Sentiment Skoru
# ─────────────────────────────────────────────────────────────────────────────

def compute_sentiment_score(text: str) -> float:
    """
    Sözlük tabanlı negatiflik / aciliyet skoru.
    0 = nötr, 1 = maksimum negatif sinyal.

    BERT yerine bu yöntemi tercih etmemizin nedeni:
    Sonuçların SHAP grafiğinde kelime bazında açıklanabilmesi.
    """
    lower = text.lower()
    raw   = sum(w for kw, w in NEGATIVE_KEYWORDS.items() if kw in lower)
    return round(min(raw / NLP_SATURATION, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSON → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def parse_issues_to_dataframe(raw: list[dict], modules: list) -> pd.DataFrame:
    """
    Ham GitHub JSON listesini analitik DataFrame'e dönüştürür.
    Her satır: modül adı, issue id, tarih, yorum sayısı, sentiment skoru.
    """
    rows = []
    for issue in raw:
        full_text = (issue.get("title", "") + " " + (issue.get("body") or ""))

        # Modül ismini metin içinden çıkar
        found_mod = next((m for m in modules if m in full_text), None)
        if not found_mod:
            found_mod = np.random.choice(modules)   # Bulunamazsa rastgele ata

        rows.append({
            "module_name"    : found_mod,
            "issue_id"       : issue["id"],
            "created_at"     : pd.to_datetime(issue["created_at"]),
            "comments"       : issue.get("comments", 0),
            "issue_sentiment": compute_sentiment_score(full_text),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Modül Bazlı Agregasyon
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_module(issues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Issue DataFrame'ini modül bazında özetler.
    Çıktı: issue_count, avg_issue_sentiment, total_comments (modül başına)
    """
    return (
        issues_df
        .groupby("module_name")
        .agg(
            issue_count         =("issue_id",        "count"),
            avg_issue_sentiment =("issue_sentiment",  "mean"),
            total_comments      =("comments",         "sum"),
        )
        .reset_index()
        .round(4)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Herkese Açık Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge_issues(promise_df: pd.DataFrame) -> pd.DataFrame:
    """
    GitHub Issues verisini çeker/simüle eder ve Promise dataset ile birleştirir.

    Parametreler
    ------------
    promise_df : load_promise_dataset() çıktısı

    Dönüş
    -----
    Birleştirilmiş DataFrame — tüm sonraki adımların girdisi.
    """
    token = os.getenv("GITHUB_TOKEN", "")

    if token:
        print("🔑 GitHub Token bulundu → Gerçek API çağrısı yapılıyor...")
        raw_issues = fetch_github_issues_real(
            GITHUB_REPO_OWNER, GITHUB_REPO_NAME,
            GITHUB_ISSUE_LABEL, GITHUB_MAX_PAGES,
        )
        source = "GitHub API (gerçek)"
    else:
        print("🔁 GitHub Token yok → Simüle edilmiş veri kullanılıyor.")
        raw_issues = simulate_github_issues(GITHUB_SIM_COUNT, MODULES)
        source = "Simülasyon (API mock)"

    print(f"   Kaynak: {source}  |  Issue sayısı: {len(raw_issues):,}")

    issues_df        = parse_issues_to_dataframe(raw_issues, MODULES)
    module_stats     = aggregate_by_module(issues_df)

    # ── Promise + GitHub Issues birleştir ────────────────────────────────────
    merged = promise_df.merge(module_stats, on="module_name", how="left")

    # Issue'su olmayan modüller → sıfır sinyal (NaN bırakma)
    merged["issue_count"]         = merged["issue_count"].fillna(0).astype(int)
    merged["avg_issue_sentiment"] = merged["avg_issue_sentiment"].fillna(0.0)
    merged["total_comments"]      = merged["total_comments"].fillna(0).astype(int)

    print(f"✅ Merge tamamlandı  |  Satır: {len(merged):,}  |"
          f"  Sütun: {len(merged.columns)}  |  Eksik değer: {merged.isnull().sum().sum()}")
    return merged
