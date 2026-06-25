# Veri Kaynakları

## Kaynak 1 — NASA JM1 Defect Dataset (Yapılandırılmış Veri)

- **Erişim Linki:** https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction
- **Dosya Adı:** jm1.csv
- **Kaynak Kurum:** NASA IV&V Facility / Promise Software Engineering Repository
- **Satır Sayısı:** 10.885 yazılım modülü
- **Sütun Sayısı:** 22 (Halstead metrikleri, McCabe karmaşıklığı, defect etiketi)
- **Lisans:** Açık erişim — akademik kullanım için serbesttir
- **Projede Kullanımı:** Ana yapılandırılmış veri kaynağı. `loc`, `v(g)` (cyclomatic complexity),
  `iv(g)` (coupling), `d` (Halstead difficulty) ve `defects` (hedef değişken) sütunları kullanılmıştır.

---

## Kaynak 2 — GitHub Issues API (Yapılandırılmamış / NLP Verisi)

- **Erişim Linki:** https://api.github.com/repos/eclipse-platform/eclipse.platform/issues
- **API Dokümantasyonu:** https://docs.github.com/en/rest/issues/issues
- **Repository:** eclipse-platform / eclipse.platform (açık kaynak)
- **Çekilen Issue Sayısı:** 205 adet (kapalı, "bug" etiketli)
- **Erişim Yöntemi:** GitHub REST API v3 — Personal Access Token ile kimlik doğrulama
- **Lisans:** GitHub API Terms of Service kapsamında serbesttir
- **Projede Kullanımı:** Her issue başlığı ve açıklaması NLP (sözlük tabanlı sentiment analizi)
  ile işlenmiş, 0–1 arası negatiflik/aciliyet skoru üretilmiştir. Bu skor `issue_sentiment_score`
  özelliği olarak modele dahil edilmiştir.

---

## Veri Füzyonu Yöntemi

İki kaynak `module_name` anahtar sütunu üzerinden `pandas.merge()` (left join) ile
birleştirilmiştir. GitHub Issues verisinden modül başına şu agregat değerler hesaplanmıştır:

- `issue_count` — Modülün aldığı toplam bug raporu sayısı
- `avg_issue_sentiment` — Ortalama NLP negatiflik skoru
- `total_comments` — Toplam yorum sayısı (ilgi yoğunluğu sinyali)

Issue kaydı bulunmayan modüller için bu değerler sıfır olarak atanmıştır (NaN bırakılmamıştır).
