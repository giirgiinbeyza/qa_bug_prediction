"""
config.py
─────────
Projenin tüm sabitleri burada tanımlanır.
Bir değeri değiştirmek istediğinde tek bir dosyaya bakman yeterli.
"""

import numpy as np

# ── Proje Genel ──────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_MODULES    = 1_500        # Simüle edilecek modül sayısı

# ── Yazılım Modül İsimleri ───────────────────────────────────────────────────
MODULES = [
    "core.auth", "core.session", "ui.dashboard", "ui.forms",
    "api.gateway", "api.payments", "db.connector", "db.migrations",
    "utils.logger", "utils.cache", "reporting.pdf", "reporting.csv",
    "notifications.email", "notifications.push", "search.engine",
]

# ── GitHub API ────────────────────────────────────────────────────────────────
GITHUB_REPO_OWNER  = "eclipse-platform"
GITHUB_REPO_NAME   = "eclipse.platform"
GITHUB_ISSUE_LABEL = "bug"
GITHUB_MAX_PAGES   = 5
GITHUB_SIM_COUNT   = 600     # Token yoksa kaç simüle issue üretilsin

# ── NLP Negatiflik Sözlüğü ───────────────────────────────────────────────────
NEGATIVE_KEYWORDS = {
    "critical"    : 3.0,
    "crash"       : 2.5,
    "broken"      : 2.0,
    "corrupt"     : 2.5,
    "security"    : 3.0,
    "leak"        : 2.0,
    "fail"        : 1.5,
    "error"       : 1.5,
    "exception"   : 1.5,
    "null"        : 1.0,
    "wrong"       : 1.0,
    "timeout"     : 1.5,
    "regression"  : 2.0,
    "loss"        : 2.0,
    "expose"      : 2.5,
    "unstable"    : 1.5,
    "degradation" : 1.5,
    "intermittent": 1.0,
    "concurrent"  : 1.0,
}
NLP_SATURATION = 10.0   # Normalize için doyum noktası

# ── Model Hiperparametreleri ─────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"    : 400,
    "max_depth"       : 5,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.75,
    "eval_metric"     : "logloss",
    "random_state"    : RANDOM_SEED,
    "n_jobs"          : -1,
}
CV_FOLDS = 5

# ── İş Maliyetleri (TL) ──────────────────────────────────────────────────────
COST_QA_MANUAL = 500        # FP maliyeti: gereksiz yoğun test (TL/commit)
COST_HOTFIX    = 15_000     # FN maliyeti: canlıda bug → hotfix (TL/olay)

# ── Eşik Optimizasyonu ───────────────────────────────────────────────────────
THRESHOLD_RANGE = np.arange(0.01, 1.00, 0.01)

# ── Özellik Sütunları ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "loc", "cyclomatic_complexity", "num_authors", "commit_count",
    "coupling", "lack_cohesion",
    "code_churn_30d", "developer_fatigue_index",
    "historical_bug_rate", "issue_sentiment_score",
    "complexity_x_churn", "authors_per_commit", "loc_per_commit",
    "issue_count", "total_comments",
]
TARGET_COL = "has_defect"

# SHAP grafiklerinde görünecek Türkçe etiketler
FEATURE_LABELS = {
    "loc"                    : "Kod Satırı (LOC)",
    "cyclomatic_complexity"  : "Cyclomatic Karmaşıklık",
    "num_authors"            : "Yazar Sayısı",
    "commit_count"           : "Toplam Commit",
    "coupling"               : "Bağımlılık (Coupling)",
    "lack_cohesion"          : "Kohezyon Eksikliği",
    "code_churn_30d"         : "Code Churn (30 gün)",
    "developer_fatigue_index": "Geliştirici Yorgunluğu",
    "historical_bug_rate"    : "Geçmiş Bug Oranı",
    "issue_sentiment_score"  : "NLP Şikayet Skoru",
    "complexity_x_churn"     : "Karmaşıklık × Churn",
    "authors_per_commit"     : "Commit Başına Yazar",
    "loc_per_commit"         : "Commit Başına Satır",
    "issue_count"            : "Issue Sayısı",
    "total_comments"         : "Toplam Yorum",
}

# ── Görsel Tema ───────────────────────────────────────────────────────────────
PLOT_THEME = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor"  : "#1A1D27",
    "axes.edgecolor"  : "#3A3D4D",
    "axes.labelcolor" : "#E0E0E0",
    "xtick.color"     : "#A0A0A0",
    "ytick.color"     : "#A0A0A0",
    "text.color"      : "#E0E0E0",
    "grid.color"      : "#2A2D3D",
    "grid.linewidth"  : 0.6,
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
}

ACCENT  = "#00D4FF"   # Cyan
DANGER  = "#FF4C6A"   # Kırmızı
SUCCESS = "#00C896"   # Yeşil
WARNING = "#FFB347"   # Turuncu
NEUTRAL = "#7C83FD"   # Mor

# ── Çıktı Dizini ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
