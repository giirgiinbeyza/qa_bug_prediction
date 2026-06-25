# qa_bug_prediction
# QA Bug Tahmini & Test Eforu Optimizasyonu

Yazılım kalite güvencesinde makine öğrenmesi ile maliyet odaklı bug tahmini.

## Proje Özeti

Bir yazılım projesindeki kod değişikliklerinin (commit) production ortamına
çıkmadan önce hata barındırıp barındırmadığını tahmin eden, maliyet odaklı
bir XGBoost modeli. Standart %50 eşik yerine iş maliyetlerine (QA: 500 TL /
Hotfix: 15.000 TL) dayalı optimal karar eşiği kullanılmaktadır.

## Veri Kaynakları

- **NASA JM1 Defect Dataset** (Kaggle) — 10.885 gerçek yazılım modülü
- **GitHub Issues API** — Eclipse reposundan çekilen gerçek bug raporları

Detaylar için `Veri_Kaynaklari.md` dosyasına bakınız.

## Kullanılan Teknolojiler

Python · XGBoost · SHAP · Scikit-learn · Pandas · NumPy · GitHub REST API

## Proje Yapısı

```
data/         → Veri yükleme ve füzyon kodları
features/     → Özellik mühendisliği
model/        → Model eğitimi ve maliyet optimizasyonu
reporting/    → EDA, SHAP ve ROI görselleştirmeleri
QA_Bug_Prediction.ipynb  → Ana analiz notebook'u
```

## Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook QA_Bug_Prediction.ipynb
```

> GitHub API'den gerçek veri çekmek için `GITHUB_TOKEN` ortam değişkeni
> tanımlanmalıdır. Tanımlı değilse sistem otomatik olarak simülasyon
> moduna geçer.

## Sonuçlar

- CV ROC-AUC: 0.699
- Optimal karar eşiği: 0.02
- Standart eşiğe göre maliyet tasarrufu: %74,9

---
BEYZA GİRGİN. JR.QA Engineer
