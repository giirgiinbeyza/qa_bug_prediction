"""
reporting/shap_explainer.py
────────────────────────────
SHAP ile modelin kararlarını açıklar.
"Model bu commit'te neden yüksek risk gördü?" sorusunu sayısal olarak yanıtlar.
"""

import os
import shap
import numpy  as np
import matplotlib.pyplot as plt

from config import FEATURE_COLS, FEATURE_LABELS, OUTPUT_DIR, ACCENT, PLOT_THEME


def run_shap_analysis(model, X, y_prob, df, opt_threshold: float, save: bool = True):
    """
    SHAP analizi yapar, özet grafiği kaydeder ve en riskli 3 commit'i açıklar.

    Parametreler
    ------------
    model         : Eğitilmiş XGBClassifier
    X             : Özellik matrisi
    y_prob        : Bug olasılıkları
    df            : Orijinal merged DataFrame (modül adı için)
    opt_threshold : Optimal karar eşiği
    """
    plt.rcParams.update(PLOT_THEME)

    # ── SHAP Değerleri ────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Türkçe etiketli kopya
    X_labeled = X.rename(columns=FEATURE_LABELS)

    # ── Özet Bar Grafiği ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    shap.summary_plot(shap_values, X_labeled, plot_type="bar",
                      show=False, color=ACCENT)
    plt.title("SHAP Özellik Önem Sıralaması\n(Bug Olasılığına Katkı)",
              fontsize=13, color="#E0E0E0", pad=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "shap_importance.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"   📊 SHAP önem grafiği → {path}")

    # ── Yüksek Riskli Commit Açıklamaları ────────────────────────────────────
    risky_indices = np.where(y_prob >= opt_threshold)[0]
    if len(risky_indices) == 0:
        print("   ⚠️  Optimal eşiği aşan commit bulunamadı.")
        return shap_values

    top3 = risky_indices[np.argsort(y_prob[risky_indices])[-3:]][::-1]

    print("\n🔴 YÜKSEK RİSKLİ COMMİT AÇIKLAMALARI (QA Lead Raporu):")
    for rank, idx in enumerate(top3, 1):
        prob  = y_prob[idx]
        row   = df.iloc[idx]
        sv    = shap_values[idx]
        top_f = sorted(zip(sv, FEATURE_COLS), key=lambda x: abs(x[0]), reverse=True)[:3]

        print(f"\n  [{rank}] Modül: {row['module_name']:<22} | Bug Olasılığı: {prob:.1%}")
        print(f"       Model bu commit'te yüksek risk gördü çünkü:")
        for shap_val, feat in top_f:
            direction = "📈 artırdı" if shap_val > 0 else "📉 azalttı"
            label     = FEATURE_LABELS.get(feat, feat)
            val       = row[feat]
            print(f"       → {label:<30} = {val:>8.2f}"
                  f"  (SHAP: {shap_val:+.3f}) → riski {direction}")

    return shap_values
