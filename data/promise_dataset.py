"""
data/promise_dataset.py
───────────────────────
NASA JM1 Defect Dataset'i okur ve projenin beklediği formata dönüştürür.

Kaynak : https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction
Dosya  : jm1.csv  (data/ klasörüne koyulmalı)

JM1 Hakkında
────────────
NASA'nın gerçek bir yazılım projesinden derlenen defect dataset'i.
10.000+ modül, 21 kod metriği ve bir hedef değişken (defects: true/false) içerir.
Halstead metrikleri, McCabe karmaşıklığı ve satır sayıları bulunur.
"""

import os
import numpy  as np
import pandas as pd
from config import RANDOM_SEED, MODULES

CSV_PATH = os.path.join(os.path.dirname(__file__), "jm1.csv")


def load_promise_dataset() -> pd.DataFrame:
    """
    JM1 CSV'sini okur, temizler ve projenin beklediği sütun isimlerine dönüştürür.

    Dönüş sütunları
    ---------------
    module_name            : Yapay modül ismi (JM1'de yok, config'den atanır)
    loc                    : Satır sayısı
    cyclomatic_complexity  : McCabe v(g)
    num_authors            : JM1'de yok → LOC ile korelasyonlu üretilir
    commit_count           : JM1'de yok → LOC ile korelasyonlu üretilir
    coupling               : iv(g) — internal cyclomatic complexity
    lack_cohesion          : Halstead difficulty (d) normalize edilmiş hali
    has_defect             : Hedef değişken (0/1)
    """

    # ── CSV oku ───────────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"\n❌ '{CSV_PATH}' bulunamadı.\n"
            f"   jm1.csv dosyasını data/ klasörüne koyun.\n"
            f"   İndirme: https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction"
        )

    raw = pd.read_csv(CSV_PATH)
    print(f"📦 JM1 CSV okundu  |  Satır: {len(raw):,}  |  Sütun: {len(raw.columns)}")

    # Sütun isimlerini normalize et
    raw.columns = raw.columns.str.strip().str.lower()

    # ── Hedef değişken ────────────────────────────────────────────────────────
    if "defects" not in raw.columns:
        raise ValueError(f"'defects' sütunu bulunamadı. Mevcut: {list(raw.columns)}")

    raw["has_defect"] = raw["defects"].apply(
        lambda x: 1 if str(x).strip().lower() in ["true", "1", "yes"] else 0
    )

    # ── Eksik değer temizliği ─────────────────────────────────────────────────
    raw   = raw.replace("?", np.nan)
    before = len(raw)
    raw   = raw.dropna(subset=["loc", "v(g)", "has_defect"])
    after  = len(raw)
    if before - after > 0:
        print(f"   Eksik değer temizlendi: {before - after} satır düşürüldü.")

    # Sayısal dönüşüm
    for col in ["loc", "v(g)", "ev(g)", "iv(g)", "d", "branchcount"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["loc", "v(g)"]).reset_index(drop=True)

    # ── Modül ismi ────────────────────────────────────────────────────────────
    np.random.seed(RANDOM_SEED)
    module_names = np.random.choice(MODULES, len(raw))

    # ── num_authors & commit_count ─────────────────────────────────────────────
    # JM1'de bu bilgiler yok.
    # Endüstri gerçeği: büyük modüllere daha fazla geliştirici dokunur.
    np.random.seed(RANDOM_SEED + 10)
    loc_norm     = (raw["loc"] / raw["loc"].max()).values
    num_authors  = np.clip(np.random.poisson(1 + loc_norm * 8),  1,  12).astype(int)
    commit_count = np.clip(np.random.poisson(5 + loc_norm * 40), 1, 200).astype(int)

    # ── Lack of cohesion ──────────────────────────────────────────────────────
    if "d" in raw.columns:
        d_vals        = raw["d"].fillna(0).values
        lack_cohesion = np.clip(d_vals / (d_vals.max() + 1e-9), 0, 1)
    else:
        lack_cohesion = np.random.beta(1.5, 4, len(raw))

    # ── Coupling ──────────────────────────────────────────────────────────────
    coupling = raw["iv(g)"].fillna(raw["v(g)"]).values \
               if "iv(g)" in raw.columns else raw["v(g)"].values

    # ── Final DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "module_name"          : module_names,
        "loc"                  : raw["loc"].values.astype(int),
        "cyclomatic_complexity": np.round(raw["v(g)"].values, 2),
        "num_authors"          : num_authors,
        "commit_count"         : commit_count,
        "coupling"             : np.round(coupling, 2),
        "lack_cohesion"        : np.round(lack_cohesion, 4),
        "has_defect"           : raw["has_defect"].values.astype(int),
    })

    # Aykırı değerleri clip et
    df["loc"]                   = df["loc"].clip(1, 50_000)
    df["cyclomatic_complexity"] = df["cyclomatic_complexity"].clip(1, 200)
    df["coupling"]              = df["coupling"].clip(0, 100)

    print(f"   Temizlenmiş veri  |  Satır: {len(df):,}  |  Bug oranı: {df['has_defect'].mean():.1%}")
    return df