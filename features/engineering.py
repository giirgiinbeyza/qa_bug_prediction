"""
features/engineering.py
───────────────────────
Ham veriden iş mantığına dayalı yeni özellikler üretir.

Her fonksiyon tek bir özelliği hesaplar → test edilebilir, değiştirilebilir.
"""

import numpy  as np
import pandas as pd
from config import N_MODULES, RANDOM_SEED


def add_code_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    code_churn_30d — Son 30 gündeki commit yoğunluğu tahmini.

    Neden önemli: Çok kurcalanan kod her zaman daha fazla bug üretir.
    Gerçek projede: git log --after="30 days ago" -- <file> | wc -l
    """
    np.random.seed(RANDOM_SEED + 1)
    df["code_churn_30d"] = np.round(
        (df["commit_count"] * np.random.uniform(0.05, 0.30, len(df)))
        .clip(0, 50)
        .astype(int)
        .astype(float),
        1,
    )
    return df


def add_developer_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    developer_fatigue_index — Geliştirici yorgunluk indeksi (0.0 – 1.0).

    0.0 = Mesai saatinde, normal iş günü
    1.0 = Cuma akşamı 18:00+ veya gece yarısı veya hafta sonu

    Gerçek projede: git log --format="%ad" ile commit saati/günü parse edilir.
    """
    np.random.seed(RANDOM_SEED + 2)

    hour_probs = np.array([0.01]*6 + [0.04]*6 + [0.06]*6 + [0.03]*6, dtype=float)
    hour_probs /= hour_probs.sum()
    commit_hours = np.random.choice(range(24), len(df), p=hour_probs)

    day_probs = np.array([0.18, 0.20, 0.22, 0.20, 0.15, 0.03, 0.02], dtype=float)
    day_probs /= day_probs.sum()
    commit_weekdays = np.random.choice(range(7), len(df), p=day_probs)

    hour_risk = np.where(commit_hours < 7,   0.9,    # Gece (00-06)
                np.where(commit_hours > 18,  0.7,    # Akşam (19-23)
                np.where(commit_hours > 17,  0.5,    # Erken akşam (18)
                                             0.1)))  # Normal mesai

    day_risk = np.where(commit_weekdays >= 5, 0.8,   # Hafta sonu
               np.where(commit_weekdays == 4, 0.4,   # Cuma
                                              0.0))  # Haftaiçi

    df["developer_fatigue_index"] = np.round((hour_risk + day_risk).clip(0, 1), 4)
    df["_commit_hour"]    = commit_hours      # SHAP yorumu için sakla
    df["_commit_weekday"] = commit_weekdays
    return df


def add_historical_bug_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    historical_bug_rate — Modülün geçmişteki bug / commit oranı.

    Gerçek projede: Jira ticket'ları veya commit history'den hesaplanır.
    """
    bug_history = (
        df.groupby("module_name")["has_defect"]
        .agg(total_bugs="sum", total_commits="count")
        .assign(historical_bug_rate=lambda x: (x["total_bugs"] / x["total_commits"]).round(4))
        .reset_index()
    )
    df = df.merge(bug_history[["module_name", "historical_bug_rate"]],
                  on="module_name", how="left")
    return df


def add_issue_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    issue_sentiment_score — NLP'den gelen modül bazlı şikayet yoğunluğu.
    avg_issue_sentiment zaten Bölüm 1'de merge edildi; burada sadece yeniden adlandırıyoruz.
    """
    df["issue_sentiment_score"] = df["avg_issue_sentiment"]
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Türetilmiş etkileşim özellikleri — SHAP'ta güçlü sinyal verirler.

    complexity_x_churn  : Yüksek karmaşıklık + sık değişim = en riskli kombinasyon
    authors_per_commit  : Commit başına yazar → koordinasyon riski
    loc_per_commit      : LOC yüksek, commit az → riskli toplu merge sinyali
    """
    df["complexity_x_churn"] = np.round(
        df["cyclomatic_complexity"] * df["code_churn_30d"] / 100, 4)

    df["authors_per_commit"] = np.round(
        df["num_authors"] / (df["commit_count"] + 1), 4)

    df["loc_per_commit"] = np.round(
        df["loc"] / (df["commit_count"] + 1), 2)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon — Tüm adımları sırayla çalıştırır
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm özellik mühendisliği adımlarını pipeline olarak çalıştırır.
    main.py buraya tek satırla erişir.
    """
    df = merged_df.copy()
    df = add_code_churn(df)
    df = add_developer_fatigue(df)
    df = add_historical_bug_rate(df)
    df = add_issue_sentiment(df)
    df = add_interaction_features(df)

    print(f"✅ Özellik mühendisliği tamamlandı  |  Toplam sütun: {len(df.columns)}")
    return df
