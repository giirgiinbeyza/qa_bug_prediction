"""
reporting/eda.py
────────────────
Keşifçi veri analizi grafikleri.
Özellik dağılımları ve korelasyon matrisi.
"""

import os
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np

from config import (
    FEATURE_COLS, TARGET_COL, OUTPUT_DIR,
    SUCCESS, DANGER, ACCENT, PLOT_THEME,
)


def plot_feature_distributions(df, save: bool = True) -> str:
    """
    Bug var / yok grupları arasındaki 6 temel özelliğin yoğunluk grafiği.
    Hangi özelliğin iki grubu ayırt ettiğini görsel olarak gösterir.
    """
    plt.rcParams.update(PLOT_THEME)

    plot_pairs = [
        ("code_churn_30d",          "Code Churn (30 gün)"),
        ("developer_fatigue_index", "Developer Fatigue Index"),
        ("cyclomatic_complexity",   "Cyclomatic Complexity"),
        ("issue_sentiment_score",   "NLP Sentiment Skoru"),
        ("historical_bug_rate",     "Geçmiş Bug Oranı"),
        ("complexity_x_churn",      "Karmaşıklık × Churn"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EDA — Mühendislik Özellikleri vs Bug Varlığı",
                 fontsize=16, color="#E0E0E0", y=1.01, fontweight="bold")

    for ax, (col, title) in zip(axes.flat, plot_pairs):
        for label, clr, lbl in [(0, SUCCESS, "Temiz Kod"), (1, DANGER, "Bug Var")]:
            vals = df.loc[df[TARGET_COL] == label, col].dropna()
            ax.hist(vals, bins=30, alpha=0.65, color=clr, label=lbl, density=True)
        ax.set_title(title, fontsize=12, color="#E0E0E0")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Yoğunluk", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_distributions.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 EDA dağılım grafiği → {path}")
    return path


def plot_correlation_heatmap(df, save: bool = True) -> str:
    """
    Tüm özellikler + hedef değişken arasındaki korelasyon matrisi.
    Yüksek korelasyon → multicollinearity riski → özellik seçimi için ipucu.
    """
    plt.rcParams.update(PLOT_THEME)

    corr_cols = FEATURE_COLS + [TARGET_COL]
    corr      = df[corr_cols].corr()
    mask      = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 8},
                linewidths=0.5, linecolor="#0F1117")
    ax.set_title("Özellik Korelasyon Matrisi", fontsize=14, color="#E0E0E0", pad=15)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 Korelasyon ısı haritası → {path}")
    return path
