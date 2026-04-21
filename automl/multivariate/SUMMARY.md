# Multivariate Rashomon Analizi — Ozet

## 1. Problem Tanimi

TELCO veri setinde **TS1** zaman serisini hedef degisken olarak alip, diger 11 zaman serisini (**TS2–TS12**) bagimsiz degisken olarak kullanan tabular regresyon modelleri egitildi. Amac: farkli model ailelerinin TS1'e verdigi SHAP aciklamalarindaki tutarsizligi (Rashomon etkisi) olcmek ve bu tutarsizligin anomalilerle iliskisini incelemek.

**Onemli:** Lag, rolling mean, takvim degiskenleri gibi muhendislik ozellikleri kullanilmadi — sadece ham TS2–TS12 degerleri.

---

## 2. Veri Seti

| Parca | Donem | Zaman Adimi | Satir Sayisi |
|-------|-------|-------------|-------------|
| Train | 2021-01-01 — 2021-03-31 | 5 dk | 25,920 |
| Val   | 2021-04-01 — 2021-04-30 | 5 dk | 8,640 |
| Test  | 2021-05-01 — 2021-07-31 | 5 dk | 26,496 |

- **Hedef:** TS1
- **Ozellikler (11):** TS2, TS3, TS4, TS5, TS6, TS7, TS8, TS9, TS10, TS11, TS12
- **Anomali:** Test setinde TS1 icin 412 anomali zaman adimi (15 farkli gunde yogunlasmis)
- Anomali gunleri: 05-22, 06-02, 06-03, 06-08, 06-09, 06-10, 06-16, 06-19, 06-21, 06-24, 06-30, 07-03, 07-04, 07-06, 07-27

---

## 3. Egitilen Modeller

Toplam **10 model** 3 farkli framework kullanilarak egitildi:

### 3.1 Sklearn/Standalone (5 model)

| Model | Framework | Tur | R² | RMSE |
|-------|-----------|-----|-----|------|
| LightGBM | lightgbm | Gradient Boosting | 0.981 | 0.140 |
| XGBoost | xgboost | Gradient Boosting | 0.981 | 0.139 |
| RandomForest | sklearn | Bagging | 0.981 | 0.140 |
| Ridge | sklearn | Lineer Regresyon | 0.953 | 0.219 |
| DecisionTree | sklearn | Tek Agac | 0.978 | 0.151 |

### 3.2 H2O AutoML (5 model)

H2O AutoML 300 saniye calistirildi, 11 model egitti. En iyi model her aileden secildi:

| Model | Tur | R² | RMSE | Surrogate R² |
|-------|-----|-----|------|-------------|
| GBM | Gradient Boosting | 0.980 | 0.141 | 0.999 |
| DRF | Distributed Random Forest | 0.981 | 0.138 | 0.998 |
| XRT | Extremely Randomized Trees | 0.981 | 0.138 | 0.998 |
| DeepLearning | Yapay Sinir Agi (MLP) | 0.975 | 0.161 | 0.999 |
| GLM | Generalized Linear Model | 0.953 | 0.219 | 0.999 |

> **Not:** H2O'da XGBoost Java uyumsuzlugu nedeniyle kullanilamadi.

### 3.3 AutoGluon Tabular (6 model — onceki denemeler)

| Model | R² | RMSE |
|-------|-----|------|
| WeightedEnsemble_L2 | 0.982 | 0.137 |
| WeightedEnsemble_L3 | 0.982 | 0.137 |
| LightGBMXT_BAG_L2 | 0.981 | 0.138 |
| LightGBMXT_BAG_L1 | 0.981 | 0.139 |
| RandomForestMSE_BAG_L1 | 0.981 | 0.140 |
| LightGBM_BAG_L1 | 0.978 | 0.149 |

> AutoGluon modelleri Rashomon analizi icin yeterli cesitlilik saglamadi (hepsi tree-based), bu yuzden sklearn + H2O modelleri eklendi.

---

## 4. SHAP Hesaplama Yontemi

Modellerin cogu kara kutu oldugu icin **Surrogate TreeSHAP** yontemi kullanildi:

1. Her model icin ayri bir **LightGBM surrogate** egitildi (modelin tahminlerini taklit eden)
2. Surrogate uzerinden **TreeSHAP** ile hizli SHAP degerleri hesaplandi
3. **Faithfulness (Sadakat):** Tum surrogate'lar R² > 0.998 ile orijinal modellerin tahminlerini neredeyse birebir kopyaladi

Istisnalar:
- **Ridge:** `shap.LinearExplainer` ile dogrudan hesaplandi
- **Tree modeller (sklearn):** `shap.TreeExplainer` ile dogrudan hesaplandi

---

## 5. Rashomon Etkisi

Rashomon etkisi = ayni veriyi esit derecede iyi aciklayan farkli modellerin, farkli ozelliklere farkli onem vermesi.

### 5.1 Olcum

Her zaman adiminda, her ozellik icin:
- **Rashomon σ** = 10 modelin SHAP degerlerinin standart sapmasi
- σ yuksekse → modeller o ozelligin o andaki etkisi konusunda anlasamiyor

### 5.2 Ozellik Bazinda Rashomon σ (test seti ortalamasi)

| Ozellik | Rashomon σ | Yorum |
|---------|-----------|-------|
| TS2 | 0.132 | En yuksek belirsizlik |
| TS6 | 0.092 | Yuksek |
| TS5 | 0.067 | Orta |
| TS7 | 0.023 | Dusuk |
| TS12 | 0.023 | Dusuk |
| TS4 | 0.017 | Dusuk |
| TS11 | 0.017 | Dusuk |
| TS10 | 0.014 | Dusuk |
| TS3 | 0.013 | Dusuk |
| TS8 | 0.010 | Cok dusuk |
| TS9 | 0.003 | Cok dusuk |

> TS2 ve TS6 en dominant ozellikler ve ayni zamanda modellerin en cok anlasamadigi ozellikler. Lineer modeller (Ridge, GLM) ile tree modellerin bu ozellikleri cok farkli yorumlamasindan kaynaklaniyor.

---

## 6. SHAP–Anomali Iliskisi

### 6.1 |SHAP| Degerleri: Anomali vs Normal

Welch t-testi, Cohen's d etki buyuklugu ve noktasal iki seri korelasyonu (point-biserial r) kullanildi.

| Ozellik | Anom |SHAP| | Norm |SHAP| | Oran | Cohen's d | p-degeri | Yorum |
|---------|------------|------------|-------|----------|----------|-------|
| **TS3** | 0.0143 | 0.0106 | 1.35x | **+0.49** | 1.2e-17 *** | Anomalide ARTIYOR |
| **TS12** | 0.0121 | 0.0197 | 0.61x | **-0.46** | 9.2e-47 *** | Anomalide AZALIYOR |
| **TS11** | 0.0094 | 0.0130 | 0.73x | **-0.42** | 1.5e-18 *** | Anomalide AZALIYOR |
| **TS10** | 0.0062 | 0.0042 | 1.47x | **+0.41** | 2.4e-13 *** | Anomalide ARTIYOR |
| **TS4** | 0.0235 | 0.0184 | 1.28x | **+0.39** | 2.6e-11 *** | Anomalide ARTIYOR |
| **TS9** | 0.0047 | 0.0033 | 1.43x | **+0.30** | 3.7e-07 *** | Anomalide ARTIYOR |
| **TS5** | 0.0281 | 0.0231 | 1.21x | **+0.29** | 1.6e-06 *** | Anomalide ARTIYOR |
| **TS2** | 0.3519 | 0.3953 | 0.89x | -0.21 | 2.9e-05 *** | Hafif azaliyor |
| TS7 | 0.0050 | 0.0046 | 1.08x | +0.10 | 0.029 * | Zayif |
| TS6 | 0.5642 | 0.5575 | 1.01x | +0.02 | 0.630 ns | Fark yok |
| TS8 | 0.0074 | 0.0071 | 1.05x | +0.06 | 0.269 ns | Fark yok |

#### Cohen's d Yorumlama

| d degeri | Buyukluk |
|----------|----------|
| < 0.2 | Ihmal edilebilir |
| 0.2–0.5 | Kucuk |
| 0.5–0.8 | Orta |
| > 0.8 | Buyuk |

**Bulgu:** 8/11 ozellikte istatistiksel olarak anlamli fark var. Anomali sirasinda **TS3, TS10, TS9, TS4, TS5** daha etkili hale geliyor; **TS12, TS11** ise etkisini kaybediyor. En dominant ozellikler (TS6, TS2) ise anomaliden etkilenmiyor.

### 6.2 Rashomon σ: Anomali vs Normal

| Ozellik | σ (Anom) | σ (Norm) | Cohen's d | p-degeri |
|---------|----------|----------|----------|----------|
| **TS4** | 0.0224 | 0.0166 | **+0.41** | 1.1e-10 *** |
| **TS10** | 0.0185 | 0.0141 | **+0.39** | 4.5e-11 *** |
| **TS6** | 0.1132 | 0.0917 | **+0.37** | 4.3e-13 *** |
| **TS3** | 0.0158 | 0.0125 | **+0.33** | 3.9e-09 *** |
| **TS9** | 0.0043 | 0.0033 | **+0.32** | 2.2e-08 *** |
| **TS2** | 0.1430 | 0.1313 | **+0.30** | 5.1e-09 *** |
| TS12 | 0.0193 | 0.0228 | -0.27 | 3.3e-11 *** |
| TS5 | 0.0784 | 0.0665 | +0.22 | 3.5e-04 *** |
| TS11 | 0.0153 | 0.0174 | -0.18 | 2.9e-04 *** |
| TS7 | 0.0242 | 0.0229 | +0.09 | 0.09 ns |
| TS8 | 0.0100 | 0.0095 | +0.09 | 0.10 ns |

**Bulgu:** 10 model kullanildiginda Rashomon belirsizligi anomali sirasinda 9/11 ozellikte anlamli olarak **artiyor**. Yani anomali anlarinda modeller birbirinden daha farkli dusunuyor — bu Rashomon etkisinin anomali tespitinde bilgi tasiyabilecegini gosteriyor.

---

## 7. Anomali Siniflandirma Sonuclari

SHAP degerlerini ve Rashomon belirsizligini ozellik olarak kullanarak anomali tespiti yapildi. Zamansal bolme (%60 train, %40 test), LightGBM siniflandirici, sinif agirliklandirma (scale_pos_weight) uygulanarak.

### 7.1 Farkli Ozellik Setleri Karsilastirmasi (10 model)

| Yaklasim | Ozellik Sayisi | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|----------|---------------|---------|--------|------|-----------|--------|
| **Mean |SHAP| (11)** | 11 | **0.889** | **0.276** | **0.516** | 0.478 | **0.559** |
| All SHAP (110) | 110 | 0.866 | 0.047 | 0.133 | 0.087 | 0.288 |
| Mean + Rashomon (23) | 23 | 0.829 | 0.053 | 0.063 | 0.400 | 0.034 |
| Rashomon σ (12) | 12 | 0.776 | 0.042 | 0.105 | 0.235 | 0.068 |
| All Combined (133) | 133 | 0.651 | 0.008 | 0.019 | 0.010 | 0.119 |

### 7.2 Onceki Denemeler (3 AutoGluon model)

| Yaklasim | ROC-AUC | F1 | Precision | Recall |
|----------|---------|------|-----------|--------|
| SHAP Only (14 feat) | 0.842 | 0.607 | 0.900 | 0.458 |
| Uncertainty Only (12 feat) | 0.760 | 0.041 | 0.021 | 0.559 |
| SHAP + Uncertainty (26 feat) | 0.841 | 0.444 | 0.645 | 0.339 |

### 7.3 Cikarilar

1. **En iyi yaklasim: Mean |SHAP| (11 ozellik)** — basit ve etkili (AUC=0.889, F1=0.516)
2. Fazla ozellik eklemek overfitting'e neden oluyor (133 ozellik → AUC=0.651)
3. Rashomon belirsizligi tek basina zayif (AUC=0.776) ama anomali sinyali tasiyor
4. SHAP degerleri anomaliyi "hissediyor" — model anomali sirasinda farkli ozelliklere yaslaniyor

---

## 8. Dosya Yapisi

```
automl/multivariate/
├── models/
│   ├── TS1/                    # AutoGluon modelleri
│   ├── diverse/                # sklearn modelleri (5 pkl)
│   │   ├── LightGBM.pkl
│   │   ├── XGBoost.pkl
│   │   ├── RandomForest.pkl
│   │   ├── Ridge.pkl
│   │   └── DecisionTree.pkl
│   └── h2o/                    # H2O modelleri (5 klasor)
│       ├── GBM_grid_1_.../
│       ├── DRF_1_.../
│       ├── XRT_1_.../
│       ├── DeepLearning_1_.../
│       └── GLM_1_.../
│
├── results/
│   ├── diverse/                # sklearn SHAP + gorseller
│   │   └── shap_{model}_TS1.csv (5 adet)
│   ├── h2o/                    # H2O SHAP + gorseller
│   │   └── shap_{model}_TS1.csv (5 adet)
│   ├── combined/               # 10 model birlesik gorseller
│   │   ├── combined_rashomon_TS1_full.png
│   │   ├── combined_rashomon_TS1_zoom_*.png
│   │   ├── combined_perfeature_TS1_full.png
│   │   └── combined_perfeature_TS1_zoom_*.png
│   ├── anomaly_10models/       # 10 model anomali analizi
│   │   ├── per_feature_stats.csv
│   │   ├── classification_comparison.csv
│   │   ├── effect_sizes.png
│   │   ├── boxplots_top5.png
│   │   ├── roc_pr_comparison.png
│   │   └── metrics_bar.png
│   ├── anomaly_clf/            # 3 model SHAP siniflandirma
│   ├── anomaly_clf_uncertainty/ # Sadece belirsizlik
│   └── anomaly_clf_combined/   # SHAP vs Unc. karsilastirma
│
├── train_multivariate.py       # AutoGluon egitim
├── train_diverse.py            # sklearn egitim (5 model)
├── train_h2o.py                # H2O AutoML egitim
├── plot_combined_rashomon.py   # 10 model birlesik gorseller
├── zoom_and_correlation.py     # Zoom + korelasyon
├── shap_anomaly_classifier.py  # SHAP ile anomali siniflandirma
├── shap_uncertainty_classifier.py  # Belirsizlik ile siniflandirma
├── shap_combined_classifier.py     # SHAP + Unc. karsilastirma
└── shap_anomaly_analysis_10models.py  # 10 model tam analiz
```

---

## 9. Temel Bulgular

1. **TS2–TS12 ile TS1 tahmin edilebilir** (R² = 0.95–0.98, model turune bagli)
2. **Farkli model turleri farkli SHAP aciklamalari veriyor** — ozellikle lineer modeller (Ridge, GLM) ile tree modeller (LightGBM, XGBoost, RF) arasinda belirgin farklar var
3. **Anomali sirasinda SHAP yapisi degisiyor** — bazi ozellikler (TS3, TS10, TS9) anomalide one cikiyor, bazilari (TS12, TS11) geri cekiliyor
4. **Rashomon belirsizligi anomalide artiyor** — modeller anomali anlarinda birbirleriyle daha fazla anlasamiyor (d=0.30–0.41)
5. **SHAP degerleri anomali tespitinde kullanilabilir** — en iyi AUC=0.889 (Mean |SHAP| ile)
6. **Sadelik onemli** — 11 ozellik ile 133 ozellikten daha iyi sonuc (overfitting onlenir)
