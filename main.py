"""
main.py
───────
Tek çalıştırma noktası. Tüm pipeline buradan yönetilir.

Kullanım:
    python main.py

GitHub Token ile gerçek veri için:
    export GITHUB_TOKEN="ghp_xxxx"
    python main.py
"""

import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Proje kök dizinini path'e ekle ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_DIR, PLOT_THEME

# ── Çıktı klasörünü oluştur ───────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update(PLOT_THEME)

# ── Pipeline adımları ─────────────────────────────────────────────────────────
from data.promise_dataset      import load_promise_dataset
from data.github_issues        import load_and_merge_issues
from features.engineering      import run_feature_engineering
from model.train               import train_model
from model.cost_optimizer      import find_optimal_threshold
from reporting.eda             import plot_feature_distributions, plot_correlation_heatmap
from reporting.shap_explainer  import run_shap_analysis
from reporting.roi_dashboard   import (plot_cost_optimization, plot_roi_dashboard,
                                       print_executive_summary)


def main():
    print("=" * 60)
    print("  QA BUG TAHMİN & TEST EFORU OPTİMİZASYONU")
    print("=" * 60)

    # 1. Veri yükle
    print("\n── BÖLÜM 1: VERİ FÜZYONU ──")
    promise_df = load_promise_dataset()
    merged_df  = load_and_merge_issues(promise_df)

    # 2. Özellik mühendisliği
    print("\n── BÖLÜM 2: ÖZELLİK MÜHENDİSLİĞİ ──")
    featured_df = run_feature_engineering(merged_df)

    # 3. EDA
    print("\n── BÖLÜM 3: KEŞİFÇİ ANALİZ ──")
    plot_feature_distributions(featured_df)
    plot_correlation_heatmap(featured_df)

    # 4. Model eğitimi
    print("\n── BÖLÜM 4: MODEL EĞİTİMİ ──")
    model, X, y, y_prob, cv_scores = train_model(featured_df)

    # 5. Maliyet optimizasyonu
    print("\n── BÖLÜM 5: MALİYET OPTİMİZASYONU ──")
    opt_result = find_optimal_threshold(y, y_prob)
    plot_cost_optimization(opt_result)

    # 6. SHAP
    print("\n── BÖLÜM 6: SHAP AÇIKLANABILIRLIK ──")
    run_shap_analysis(model, X, y_prob, featured_df, opt_result.opt_threshold)

    # 7. ROI raporu
    print("\n── BÖLÜM 7: ROI RAPORU ──")
    print_executive_summary(opt_result, y, cv_scores)
    plot_roi_dashboard(opt_result, y, y_prob, cv_scores)

    # Özet
    print("\n" + "=" * 60)
    print("  TAMAMLANDI — Üretilen dosyalar:")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"   ✅ {f:<30} {size:>4} KB")


if __name__ == "__main__":
    main()
