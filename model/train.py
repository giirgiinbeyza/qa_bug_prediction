"""
model/train.py
──────────────
XGBoost modelini eğitir. İki aşamalı yaklaşım:

1. RandomizedSearchCV ile hiperparametre optimizasyonu (30 kombinasyon, 5-fold CV)
2. Bulunan en iyi parametrelerle final model eğitimi ve test seti değerlendirmesi

Neden RandomizedSearchCV?
- GridSearch tüm kombinasyonları dener → 5 parametre × ortalama 5 değer = 3.125 fit (çok yavaş)
- Randomized 30 iterasyonla önemli parametreleri yakalar, süreyi ~10 kat kısaltır
- Optuna gibi Bayesian yöntemler daha verimli ama ek bağımlılık gerektirir;
  RandomizedSearchCV scikit-learn ile gelir, kurulum gerektirmez
"""

import numpy  as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split, RandomizedSearchCV)
from sklearn.metrics         import classification_report
from xgboost                 import XGBClassifier

from config import FEATURE_COLS, TARGET_COL, RANDOM_SEED, CV_FOLDS


def train_model(df: pd.DataFrame):
    """
    Modeli eğitir ve (model, X_test, y_test, y_prob, cv_scores) döndürür.

    Adımlar
    -------
    1. Train/Test split (%80/%20, stratified)
    2. RandomizedSearchCV ile hiperparametre optimizasyonu (train seti üzerinde)
    3. En iyi parametrelerle final model → test seti değerlendirmesi
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    scale_pos_weight = int((y == 0).sum() / (y == 1).sum())
    print(f"   Sınıf dengesizliği ağırlığı (scale_pos_weight): {scale_pos_weight}")

    # ── Train / Test Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # ── Hiperparametre Arama Uzayı ────────────────────────────────────────────
    param_dist = {
        "n_estimators"    : [100, 200, 300, 400, 500],
        "max_depth"       : [3, 4, 5, 6, 7],
        "learning_rate"   : [0.01, 0.03, 0.05, 0.10, 0.15],
        "subsample"       : [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.70, 0.75, 0.8, 0.9],
        "scale_pos_weight": [scale_pos_weight],
    }

    base_model = XGBClassifier(
        eval_metric  = "logloss",
        random_state = RANDOM_SEED,
        n_jobs       = -1,
    )

    random_search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = param_dist,
        n_iter              = 30,        # 30 rastgele kombinasyon
        cv                  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                              random_state=RANDOM_SEED),
        scoring             = "roc_auc",
        random_state        = RANDOM_SEED,
        verbose             = 1,
        n_jobs              = -1,
    )

    print(f"\n🔍 Hiperparametre optimizasyonu başlıyor (30 iter × {CV_FOLDS}-fold)...")
    random_search.fit(X_train, y_train)

    print(f"\n🏆 En iyi parametreler:")
    for k, v in random_search.best_params_.items():
        print(f"   {k:<20} : {v}")
    print(f"\n📈 En iyi CV ROC-AUC : {random_search.best_score_:.4f}")

    # ── CV skorları (en iyi model üzerinden) ─────────────────────────────────
    model     = random_search.best_estimator_
    cv        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\n📈 {CV_FOLDS}-Fold CV ROC-AUC (en iyi model): "
          f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    for i, s in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {s:.4f}")

    # ── Test Seti Değerlendirmesi ──────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n📋 Sınıflandırma Raporu — Test Seti (threshold=0.50):")
    print(classification_report(y_test, y_pred,
                                 target_names=["Temiz Kod", "Bug Var"]))

    return model, X_test, y_test, y_prob, cv_scores
