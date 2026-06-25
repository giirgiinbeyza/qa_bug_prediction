"""
reporting/roi_dashboard.py
───────────────────────────
Maliyet optimizasyonu ve ROI görselleştirmeleri.
Yöneticiye sunulabilir format.
"""

import os
import numpy             as np
import matplotlib.pyplot as plt

from config import (
    COST_QA_MANUAL, COST_HOTFIX, OUTPUT_DIR, PLOT_THEME,
    ACCENT, DANGER, SUCCESS, WARNING, NEUTRAL,
)


def plot_cost_optimization(result, save: bool = True) -> str:
    """
    İki panel:
    Sol  → Eşik vs Toplam Maliyet eğrisi (optimal nokta vurgulanmış)
    Sağ  → FP Oranı vs FN Oranı trade-off grafiği
    """
    plt.rcParams.update(PLOT_THEME)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Maliyet Matrisi & Eşik Optimizasyonu",
                 fontsize=15, color="#E0E0E0", fontweight="bold")

    thr  = result.thresholds
    cost = result.total_costs

    # ── Sol panel ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(thr, [c/1000 for c in cost], color=ACCENT, linewidth=2.5, label="Toplam Maliyet")
    ax.axvline(result.opt_threshold, color=SUCCESS, linestyle="--", linewidth=2,
               label=f"Optimal Eşik = {result.opt_threshold:.2f}")
    ax.axvline(0.50, color=WARNING, linestyle=":", linewidth=2, label="Standart Eşik = 0.50")
    ax.axhline(result.ref_cost/1000, color=DANGER, linestyle="-.", linewidth=1.5,
               alpha=0.7, label=f"Ref. Maliyet = {result.ref_cost/1000:.0f}K TL")

    ax.scatter([result.opt_threshold], [result.opt_cost/1000], color=SUCCESS, s=120, zorder=5)
    ax.annotate(
        f"  ✓ {result.opt_cost/1000:.0f}K TL\n  Tasarruf: {result.savings/1000:.0f}K TL",
        xy=(result.opt_threshold, result.opt_cost/1000),
        fontsize=9, color=SUCCESS,
        xytext=(result.opt_threshold + 0.05, result.opt_cost/1000 + 50),
    )
    ax.fill_between(thr,
                    [result.ref_cost/1000] * len(thr),
                    [c/1000 for c in cost],
                    where=[c < result.ref_cost for c in cost],
                    alpha=0.15, color=SUCCESS, label="Tasarruf Bölgesi")

    ax.set_xlabel("Karar Eşiği (Threshold)", fontsize=12)
    ax.set_ylabel("Toplam Maliyet (×1000 TL)", fontsize=12)
    ax.set_title("Şirket Maliyeti vs Karar Eşiği", fontsize=13, color="#E0E0E0")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Sağ panel ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(thr, [r*100 for r in result.fp_rates],
             color=WARNING, linewidth=2, label="Gereksiz QA Oranı (FPR)")
    ax2.plot(thr, [r*100 for r in result.fn_rates],
             color=DANGER,  linewidth=2, label="Kaçırılan Bug Oranı (FNR)")
    ax2.axvline(result.opt_threshold, color=SUCCESS, linestyle="--", linewidth=2,
                label=f"Optimal Eşik = {result.opt_threshold:.2f}")
    ax2.axvline(0.50, color=NEUTRAL, linestyle=":", linewidth=1.5, label="Standart 0.50")

    ax2.set_xlabel("Karar Eşiği", fontsize=12)
    ax2.set_ylabel("Oran (%)", fontsize=12)
    ax2.set_title("FP Oranı vs FN Oranı (Trade-off)", fontsize=13, color="#E0E0E0")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cost_optimization.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 Maliyet optimizasyon grafiği → {path}")
    return path


def plot_roi_dashboard(result, y, y_prob, cv_scores, save: bool = True) -> str:
    """
    Üç panelli yönetici özeti:
    A → Strateji maliyet karşılaştırması (bar)
    B → Commit sınıflandırması (pasta)
    C → Yıllık kümülatif tasarruf projeksiyonu
    """
    plt.rcParams.update(PLOT_THEME)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Yönetici Özeti — QA Optimizasyon ROI Panosu",
                 fontsize=15, color="#E0E0E0", fontweight="bold", y=1.02)

    # ── [A] Maliyet Karşılaştırma ─────────────────────────────────────────────
    ax = axes[0]
    cats   = ["Tümünü\nYoğun QA", "Standart\nEşik (0.50)", f"Optimal\nEşik ({result.opt_threshold:.2f})"]
    values = [result.baseline_cost/1000, result.ref_cost/1000, result.opt_cost/1000]
    colors = [DANGER, WARNING, SUCCESS]
    bars   = ax.bar(cats, values, color=colors, width=0.5, edgecolor="#0F1117", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{val:.0f}K TL", ha="center", va="bottom", fontsize=10, color="#E0E0E0")
    ax.set_ylabel("Toplam Maliyet (×1000 TL)", fontsize=11)
    ax.set_title("Strateji Karşılaştırması", fontsize=12, color="#E0E0E0")
    ax.grid(axis="y", alpha=0.3)

    # ── [B] Risk Dağılımı Pasta ───────────────────────────────────────────────
    ax2 = axes[1]
    risk_high = (y_prob >= result.opt_threshold).sum()
    risk_low  = len(y_prob) - risk_high
    wedges, texts, autotexts = ax2.pie(
        [risk_high, risk_low],
        labels=["Yoğun QA\n(Yüksek Risk)", "Otomasyon\n(Düşük Risk)"],
        autopct="%1.1f%%", startangle=90,
        colors=[DANGER, SUCCESS],
        wedgeprops=dict(edgecolor="#0F1117", linewidth=2),
        textprops=dict(color="#E0E0E0", fontsize=10),
    )
    for at in autotexts:
        at.set_color("#0F1117")
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax2.set_title(f"Commit Sınıflandırması\n(Eşik = {result.opt_threshold:.2f})",
                  fontsize=12, color="#E0E0E0")

    # ── [C] Kümülatif Tasarruf ────────────────────────────────────────────────
    ax3 = axes[2]
    months         = np.arange(1, 13)
    monthly_saving = result.savings / 3      # 3 aylık sprint baz
    cumulative     = np.cumsum([monthly_saving] * 12)
    ax3.fill_between(months, cumulative/1000, alpha=0.25, color=SUCCESS)
    ax3.plot(months, cumulative/1000, color=SUCCESS, linewidth=2.5, marker="o", markersize=5)
    ax3.set_xlabel("Ay", fontsize=11)
    ax3.set_ylabel("Kümülatif Tasarruf (×1000 TL)", fontsize=11)
    ax3.set_title("Yıllık Kümülatif Tasarruf Projeksiyonu", fontsize=12, color="#E0E0E0")
    ax3.set_xticks(months)
    ax3.grid(True, alpha=0.3)
    ax3.annotate(f"  Yıl sonu:\n  {cumulative[-1]/1000:.0f}K TL",
                 xy=(12, cumulative[-1]/1000), fontsize=10, color=SUCCESS)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roi_dashboard.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 ROI panosu → {path}")
    return path


def print_executive_summary(result, y, cv_scores):
    """Terminal çıktısı — yönetici özeti tablosu."""
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │              YÖNETİCİ ÖZETİ — QA ROI RAPORU            │
  ├─────────────────────────────────────────────────────────┤
  │  Analiz edilen commit sayısı      : {len(y):>6,}              │
  │  Bug içeren commit oranı          : {y.mean():>6.1%}              │
  │                                                         │
  │  Model Performansı                                      │
  │  CV ROC-AUC (5-fold)              : {cv_scores.mean():>6.4f}              │
  │  Optimal Karar Eşiği              : {result.opt_threshold:>6.2f}              │
  │                                                         │
  │  Maliyet Karşılaştırması                                │
  │  Standart Eşik (0.50) Maliyet     : {result.ref_cost:>9,.0f} TL       │
  │  Optimal Eşik Maliyeti            : {result.opt_cost:>9,.0f} TL       │
  │  Dönem Tasarrufu                  : {result.savings:>9,.0f} TL       │
  │  Tasarruf Oranı                   : {result.savings_pct:>6.1%}              │
  │  Yıllık Projeksiyon               : {result.annual_savings:>9,.0f} TL       │
  │                                                         │
  │  Kaçırılan Bug @ Optimal Eşik     : {result.opt_fn:>6}              │
  │  Engellenen Hotfix Sayısı         : {result.opt_tp:>6}              │
  └─────────────────────────────────────────────────────────┘
""")
