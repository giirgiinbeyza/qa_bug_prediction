"""
model/cost_optimizer.py
────────────────────────
Standart 0.50 eşiği yerine iş maliyetlerini minimize eden optimal eşiği bulur.

Formül:
    Toplam Maliyet = (Yanlış Pozitif × COST_QA_MANUAL) + (Yanlış Negatif × COST_HOTFIX)

    FP → Temiz kodu haksız yere yoğun QA'ya göndermek     = 500 TL israf
    FN → Buglu kodu canlıya kaçırmak → hotfix + kesinti    = 15.000 TL zarar
"""

import numpy  as np
import pandas as pd
from dataclasses import dataclass
from config import COST_QA_MANUAL, COST_HOTFIX, THRESHOLD_RANGE


@dataclass
class OptimizationResult:
    """Eşik optimizasyonunun tüm çıktılarını taşır."""
    opt_threshold : float
    opt_cost      : float
    ref_cost      : float        # 0.50 eşiğindeki maliyet
    baseline_cost : float        # Tüm commit'leri QA'dan geçirmenin maliyeti
    savings       : float        # ref_cost - opt_cost
    savings_pct   : float
    annual_savings: float

    opt_tp : int
    opt_fp : int
    opt_fn : int
    opt_tn : int

    thresholds  : np.ndarray
    total_costs : list
    fp_rates    : list
    fn_rates    : list


def find_optimal_threshold(y_true, y_prob) -> OptimizationResult:
    """
    0.01'den 0.99'a her eşiği dener, toplam maliyeti minimize eden noktayı döndürür.
    """
    n = len(y_true)

    # ── Baseline & Referans maliyetleri ──────────────────────────────────────
    baseline_cost = n * COST_QA_MANUAL          # Tüm commit'leri yoğun QA'ya gönder

    ref_pred = (y_prob >= 0.50).astype(int)
    ref_fp   = int(((ref_pred == 1) & (y_true == 0)).sum())
    ref_fn   = int(((ref_pred == 0) & (y_true == 1)).sum())
    ref_cost = ref_fp * COST_QA_MANUAL + ref_fn * COST_HOTFIX

    # ── Eşik tarama döngüsü ───────────────────────────────────────────────────
    total_costs, fp_rates, fn_rates = [], [], []

    for thr in THRESHOLD_RANGE:
        pred = (y_prob >= thr).astype(int)
        tp   = int(((pred == 1) & (y_true == 1)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        fn   = int(((pred == 0) & (y_true == 1)).sum())
        tn   = int(((pred == 0) & (y_true == 0)).sum())

        cost = fp * COST_QA_MANUAL + fn * COST_HOTFIX
        total_costs.append(cost)
        fp_rates.append(fp / max(fp + tn, 1))
        fn_rates.append(fn / max(tp + fn, 1))

    # ── Optimal nokta ─────────────────────────────────────────────────────────
    opt_idx = int(np.argmin(total_costs))
    opt_thr = THRESHOLD_RANGE[opt_idx]
    opt_cost= total_costs[opt_idx]

    opt_pred = (y_prob >= opt_thr).astype(int)
    opt_tp   = int(((opt_pred == 1) & (y_true == 1)).sum())
    opt_fp   = int(((opt_pred == 1) & (y_true == 0)).sum())
    opt_fn   = int(((opt_pred == 0) & (y_true == 1)).sum())
    opt_tn   = int(((opt_pred == 0) & (y_true == 0)).sum())

    savings     = ref_cost - opt_cost
    savings_pct = savings / ref_cost if ref_cost > 0 else 0.0

    # Yıllık projeksiyon: 4 sprint/yıl
    annual_savings = savings * 4

    result = OptimizationResult(
        opt_threshold  = opt_thr,
        opt_cost       = opt_cost,
        ref_cost       = ref_cost,
        baseline_cost  = baseline_cost,
        savings        = savings,
        savings_pct    = savings_pct,
        annual_savings = annual_savings,
        opt_tp         = opt_tp,
        opt_fp         = opt_fp,
        opt_fn         = opt_fn,
        opt_tn         = opt_tn,
        thresholds     = THRESHOLD_RANGE,
        total_costs    = total_costs,
        fp_rates       = fp_rates,
        fn_rates       = fn_rates,
    )

    _print_summary(result)
    return result


def _print_summary(r: OptimizationResult):
    print(f"\n💡 Optimal Karar Eşiği : {r.opt_threshold:.2f}")
    print(f"   Optimal Maliyet     : {r.opt_cost:>10,.0f} TL")
    print(f"   Referans (0.50)     : {r.ref_cost:>10,.0f} TL")
    print(f"   Tasarruf            : {r.savings:>10,.0f} TL  ({r.savings_pct:.1%})")
    print(f"\n   Confusion Matrix @ threshold={r.opt_threshold:.2f}:")
    print(f"   TP (Doğru Bug Tespiti) : {r.opt_tp:>4}")
    print(f"   FP (Gereksiz Yoğun QA) : {r.opt_fp:>4}  ×  {COST_QA_MANUAL:,} TL"
          f"  =  {r.opt_fp * COST_QA_MANUAL:>10,.0f} TL")
    print(f"   FN (Kaçırılan Bug!)    : {r.opt_fn:>4}  ×  {COST_HOTFIX:,} TL"
          f"  =  {r.opt_fn * COST_HOTFIX:>10,.0f} TL")
    print(f"   TN (Doğru Serbest)     : {r.opt_tn:>4}")
