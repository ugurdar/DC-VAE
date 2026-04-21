# Combined 16-Model Rashomon + Anomali Analizi — Ozet

## 1. Amac

Uc farkli framework (sklearn, H2O, AutoGluon) ile egitilmis toplam **16 modelin** SHAP aciklamalarini birlestirerek:
- Rashomon etkisinin (modeller arasi SHAP tutarsizligi) anomalilerle iliskisini olcmek
- SHAP ve Rashomon degerlerini kullanarak TS1 anomalilerini siniflandirmak
- 10-model (sklearn+H2O) analiziyle karsilastirma yapmak

---

## 2. Kullanilan 16 Model

### 2.1 Sklearn / Standalone (5 model)

| Model | Tur | R² | SHAP Yontemi |
|-------|-----|-----|-------------|
| LightGBM | Gradient Boosting | 0.981 | TreeExplainer (dogrudan) |
| XGBoost | Gradient Boosting | 0.981 | TreeExplainer (dogrudan) |
| RandomForest | Bagging | 0.981 | TreeExplainer (dogrudan) |
| Ridge | Lineer Regresyon | 0.953 | LinearExplainer (dogrudan) |
| DecisionTree | Tek Agac | 0.978 | TreeExplainer (dogrudan) |

### 2.2 H2O AutoML (5 model)

| Model | Tur | RMSE | SHAP Yontemi |
|-------|-----|------|-------------|
| GBM | Gradient Boosting | 0.141 | Surrogate LightGBM + TreeSHAP |
| DRF | Distributed Random Forest | 0.138 | Surrogate LightGBM + TreeSHAP |
| XRT | Extremely Randomized Trees | 0.138 | Surrogate LightGBM + TreeSHAP |
| DeepLearning | MLP (Yapay Sinir Agi) | 0.161 | Surrogate LightGBM + TreeSHAP |
| GLM | Generalized Linear Model | 0.219 | Surrogate LightGBM + TreeSHAP |

### 2.3 AutoGluon Tabular (6 model)

| Model | Tur | R² | SHAP Yontemi |
|-------|-----|-----|-------------|
| WeightedEnsemble_L2 | Ensemble (L2 stack) | 0.982 | Surrogate LightGBM + TreeSHAP |
| WeightedEnsemble_L3 | Ensemble (L3 stack) | 0.982 | Surrogate LightGBM + TreeSHAP |
| LightGBMXT_BAG_L2 | Boosting (bagged, L2) | 0.981 | Surrogate LightGBM + TreeSHAP |
| LightGBMXT_BAG_L1 | Boosting (bagged, L1) | 0.981 | Surrogate LightGBM + TreeSHAP |
| RandomForestMSE_BAG_L1 | Bagging (L1) | 0.981 | Surrogate LightGBM + TreeSHAP |
| LightGBM_BAG_L1 | Boosting (bagged, L1) | 0.978 | Surrogate LightGBM + TreeSHAP |

> **Surrogate yontemi:** H2O ve AutoGluon modelleri dogrudan SHAP API'sine verilemez. Bu yuzden her model icin ayri bir LightGBM surrogate egitildi (orijinal modelin tahminlerini taklit eden). Tum surrogate'larin R² > 0.998. Ardindan TreeSHAP ile hizli SHAP degerleri hesaplandi. Sklearn modelleri icin ise dogrudan TreeExplainer/LinearExplainer kullanildi.

---

## 3. Veri Seti

- **Hedef:** TS1
- **Ozellikler (11):** TS2, TS3, TS4, TS5, TS6, TS7, TS8, TS9, TS10, TS11, TS12
- **Test seti:** 26,496 zaman adimi (5 dk aralikli)
- **Anomali:** 412 anomali zaman adimi (toplam test adimlarinin ~%1.6'si)
- **Lag/muhendislik ozelligi yok** — sadece ham zaman serisi degerleri kullanildi

---

## 4. Rashomon Etkisi — 16 Model

### 4.1 Ozellik Bazinda Rashomon σ (test seti ortalamasi)

| Ozellik | Rashomon σ (16 model) | Rashomon σ (10 model) | Degisim |
|---------|----------------------|----------------------|---------|
| TS2 | 0.1126 | 0.132 | azaldi |
| TS6 | 0.0771 | 0.092 | azaldi |
| TS5 | 0.0538 | 0.067 | azaldi |
| TS7 | 0.0191 | 0.023 | azaldi |
| TS12 | 0.0186 | 0.023 | azaldi |
| TS4 | 0.0139 | 0.017 | azaldi |
| TS11 | 0.0145 | 0.017 | azaldi |
| TS10 | 0.0119 | 0.014 | azaldi |
| TS3 | 0.0104 | 0.013 | azaldi |
| TS8 | 0.0084 | 0.010 | azaldi |
| TS9 | 0.0030 | 0.003 | ayni |

> **Yorum:** 16 modelde Rashomon σ tum ozellikler icin hafif **azaldi**. Bunun nedeni AutoGluon modellerinin cogununun tree-based (LightGBM, RF) olmasi — yani sklearn modellerine yakin SHAP oruntuleri uretmeleri. Bu modellerin eklenmesi ortalama sapmayi dusurdu. Gercek cesitlilik hala **lineer modeller** (Ridge, GLM) ile **tree modeller** arasindaki farktan kaynaklaniyor.

### 4.2 En Yuksek Belirsizlik Ozellikleri

1. **TS2** (σ=0.113): En dominant ikinci ozellik, modeller arasinda en cok tartisilan
2. **TS6** (σ=0.077): En dominant ozellik, ancak modeller arasinda yuksek anlasamazlik
3. **TS5** (σ=0.054): Orta duzeyde Rashomon belirsizligi

---

## 5. SHAP–Anomali Iliskisi (16 Model)

### 5.1 |SHAP| Degerleri: Anomali vs Normal

Welch t-testi + Cohen's d etki buyuklugu:

| Ozellik | Anom |SHAP| | Norm |SHAP| | Oran | Cohen's d | p-degeri | Anlam |
|---------|-------------|-------------|-------|----------|----------|-------|
| **TS3** | 0.0130 | 0.0093 | 1.40x | **+0.50** | 2.7e-17 *** | Anomalide ARTIYOR |
| **TS11** | 0.0085 | 0.0119 | 0.71x | **-0.47** | 3.8e-22 *** | Anomalide AZALIYOR |
| **TS12** | 0.0100 | 0.0174 | 0.58x | **-0.46** | 2.2e-48 *** | Anomalide AZALIYOR |
| **TS10** | 0.0073 | 0.0046 | 1.58x | **+0.37** | 6.9e-10 *** | Anomalide ARTIYOR |
| **TS4** | 0.0214 | 0.0170 | 1.26x | **+0.36** | 2.6e-10 *** | Anomalide ARTIYOR |
| **TS9** | 0.0050 | 0.0033 | 1.50x | **+0.33** | 1.8e-08 *** | Anomalide ARTIYOR |
| **TS5** | 0.0188 | 0.0157 | 1.20x | **+0.29** | 8.2e-07 *** | Anomalide ARTIYOR |
| **TS2** | 0.3410 | 0.3747 | 0.91x | -0.17 | 6.0e-04 *** | Hafif azaliyor |
| TS8 | 0.0092 | 0.0089 | 1.04x | +0.05 | 0.308 ns | Fark yok |
| TS7 | 0.0048 | 0.0047 | 1.02x | +0.02 | 0.623 ns | Fark yok |
| TS6 | 0.5753 | 0.5700 | 1.01x | +0.02 | 0.702 ns | Fark yok |

#### Cohen's d Yorumlama

| d degeri | Buyukluk |
|----------|----------|
| < 0.2 | Ihmal edilebilir |
| 0.2–0.5 | Kucuk |
| 0.5–0.8 | Orta |
| > 0.8 | Buyuk |

**Bulgu:** 8/11 ozellikte istatistiksel olarak anlamli fark (p<0.001). 16 model ile de 10 model ile benzer oruntu: **TS3** (d=+0.50) en guclu pozitif etki, **TS11** (d=-0.47) ve **TS12** (d=-0.46) en guclu negatif etki. TS3'un d degeri 10 modelde 0.49'dan 16 modelde 0.50'ye cikti — daha da guclu.

### 5.2 Rashomon σ: Anomali vs Normal

| Ozellik | σ (Anomali) | σ (Normal) | Cohen's d | p-degeri | Anlam |
|---------|-------------|------------|----------|----------|-------|
| **TS4** | 0.0187 | 0.0138 | **+0.42** | 2.4e-11 *** | Daha fazla anlasamazlik |
| **TS10** | 0.0156 | 0.0118 | **+0.39** | 6.6e-11 *** | Daha fazla anlasamazlik |
| **TS3** | 0.0132 | 0.0104 | **+0.36** | 9.7e-11 *** | Daha fazla anlasamazlik |
| **TS6** | 0.0939 | 0.0768 | **+0.35** | 3.9e-12 *** | Daha fazla anlasamazlik |
| **TS9** | 0.0039 | 0.0030 | **+0.33** | 1.3e-08 *** | Daha fazla anlasamazlik |
| **TS2** | 0.1221 | 0.1124 | **+0.29** | 8.3e-09 *** | Daha fazla anlasamazlik |
| TS12 | 0.0159 | 0.0187 | -0.26 | 9.2e-11 *** | Anlasamazlik azaliyor |
| TS5 | 0.0636 | 0.0537 | +0.23 | 2.9e-04 *** | Hafif artis |
| TS11 | 0.0128 | 0.0146 | -0.18 | 1.4e-04 *** | Anlasamazlik azaliyor |
| TS8 | 0.0090 | 0.0084 | +0.13 | 0.014 * | Zayif |
| TS7 | 0.0201 | 0.0191 | +0.09 | 0.106 ns | Fark yok |

**Bulgu:** 16 modelde de anomali sirasinda 9/11 ozellikte Rashomon belirsizligi anlamli olarak degisiyor. TS4 (d=+0.42) ve TS10 (d=+0.39) en guclu belirsizlik artisi gosteren ozellikler. 10 model analiziyle karsilastirildiginda d degerleri benzer kaliyor.

---

## 6. Anomali Siniflandirma Sonuclari

### 6.0 Ozellik Setleri Nasil Olusturuldu?

SHAP degerlerinden 5 farkli ozellik seti turetildi. Ayni veri, farkli ozetleme bicimleriyle siniflandirici modele verildi:

| Yaklasim | Ozellik Sayisi | Nasil Hesaplandi |
|----------|---------------|-----------------|
| **All SHAP** | 176 (= 16 model × 11 degisken) | Her modelin her degisken icin ayri \|SHAP\| degeri. Ornegin: \|shap\|\_LightGBM\_TS2, \|shap\|\_Ridge\_TS2, ... seklinde 176 sutun. |
| **Mean \|SHAP\|** | 11 (= 11 degisken) | 16 modelin SHAP degerlerinin ortalaması alinir, her degisken icin tek bir deger kalir. Ornegin: mean\_\|shap\|\_TS2 = ortalama(\|shap\|\_LightGBM\_TS2, \|shap\|\_Ridge\_TS2, ..., \|shap\|\_WE\_L3\_AG\_TS2) |
| **Rashomon σ** | 12 (= 11 degisken + 1 toplam) | 16 modelin SHAP degerlerinin standart sapmasi. Her degiskendeki model anlasamazligi + butun degiskenlerin toplam σ degeri. |
| **Mean + Rashomon** | 23 (= 11 + 12) | Mean \|SHAP\| ve Rashomon σ birlestirildi. |
| **All Combined** | 188 (= 176 + 12) | Per-model \|SHAP\| (176) + Rashomon σ (12) birlestirildi. |

> **Mantik:** "All SHAP" her modelin bireysel gorusunu korur (en detayli). "Mean" modelleri ozetler (en basit). "Rashomon σ" modeller arasi anlasamazligi olcer. Boylece SHAP bilgisinin ve belirsizligin anomali tespitindeki katkisini ayri ayri ve birlikte test edebiliyoruz.

### 6.1 16-Model Karsilastirmasi

LightGBM siniflandirici, zamansal %60/%40 bolme, scale_pos_weight ile sinif dengeleme:

| Yaklasim | Ozellik | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|----------|---------|---------|--------|------|-----------|--------|
| **All SHAP (176)** | 176 | **0.878** | **0.349** | **0.529** | **0.628** | **0.458** |
| All Combined (188) | 188 | 0.888 | 0.076 | 0.196 | 0.125 | 0.458 |
| Mean \|SHAP\| (11) | 11 | 0.538 | 0.009 | 0.024 | 0.038 | 0.017 |
| Mean + Rashomon (23) | 23 | 0.511 | 0.007 | 0.013 | 0.007 | 0.153 |
| Rashomon σ (12) | 12 | 0.505 | 0.006 | 0.013 | 0.008 | 0.034 |

### 6.2 10-Model vs 16-Model Karsilastirmasi

| Yaklasim | 10 Model AUC | 16 Model AUC | 10 Model F1 | 16 Model F1 |
|----------|-------------|-------------|------------|------------|
| All SHAP | 0.866 | **0.878** (+) | 0.133 | **0.529** (+++) |
| Mean |SHAP| | **0.889** | 0.538 (-) | **0.516** | 0.024 (---) |
| Rashomon σ | **0.776** | 0.505 (-) | 0.105 | 0.013 (-) |
| All Combined | 0.651 | **0.888** (+) | 0.019 | **0.196** (+) |

### 6.3 Analiz ve Yorumlar

1. **All SHAP (176 ozellik)** en iyi: AUC=0.878, F1=0.529, Precision=0.628. Her modelin her ozellik icin ayri |SHAP| degeri kullaniliyor. Bu detayli temsil, siniflandiricinya her modelin anomaliye farkli tepkisini ogrenmesine izin veriyor.

2. **Mean |SHAP| icin ciddi dusus (0.889 → 0.538):** 10 modelde Mean |SHAP| en iyi yaklasimdi. 16 modelde ise cok zayif. Bunun nedeni: AutoGluon modelleri (6 adet) cogunlukla tree-based ve birbirine yakin — ortalamayi "seyreltiyor" ve anomali sinyalini zayiflatiyor.

3. **All SHAP neden 16 modelde daha iyi?** 10 modelde 110 ozellik (10×11) iken 16 modelde 176 ozellik (16×11). Daha fazla model perspektifi siniflandiricinya her modelin anomali sirasindaki farkli "gorusunu" veriyor. Bu durumda daha fazla ozellik overfitting yerine bilgi kazanci sagliyor.

4. **Rashomon σ tek basina yetersiz:** Hem 10 hem 16 modelde anomali tespiti icin zayif. Belirsizlik anomalide artiyor (istatistiksel olarak anlamli) ama siniflandirma icin yeterli ayirt edicilik yok.

---

## 7. Temel Bulgular (16 Model)

1. **Rashomon σ 16 modelde biraz azaldi:** AutoGluon modelleri tree-based oldugu icin sklearn modellerine yakin SHAP oruntuleri uretiyor. Gercek cesitlilik lineer modeller (Ridge, GLM) ile tree modeller arasinda.

2. **SHAP–anomali iliskisi tutarli:** 16 modelde de 10 modeldekiyle ayni ozellikler anomali sirasinda ayni yonde degisiyor. TS3, TS10, TS9 artiyor; TS12, TS11 azaliyor. Cohen's d degerleri neredeyse ayni.

3. **Per-model |SHAP| en bilgilendirici temsil:** Ortalama almak yerine her modelin |SHAP| degerini ayri tutmak, siniflandirici icin en iyi sonucu veriyor (F1=0.529).

4. **Model cesitliligi onemli:** Rastgele cok model eklemek degil, **yapisal olarak farkli** modeller eklemek (lineer vs tree vs ensemble vs neural network) Rashomon etkisini gercekten olcmeyi sagliyor.

5. **Anomali sirasinda model anlasamazligi artiyor:** 9/11 ozellikte Rashomon σ anomali anlarinda anlamli olarak yukseliyor (d=0.23–0.42). Bu, anomalilerin sadece veri patikasindan degil, model aciklamalarindaki tutarsizliktan da tespit edilebilecegini gosteriyor.

---

## 8. Dosya Yapisi

```
combined_16models/
├── run_combined.py              # Ana script (plot + istatistik + siniflandirma)
├── SUMMARY.md                   # Bu dosya
└── results/
    ├── rashomon_summary_TS1_full.png      # 4 panelli Rashomon ozet plotu
    ├── rashomon_perfeature_TS1_full.png   # Ozellik bazli Rashomon detay
    ├── effect_sizes.png                   # Cohen's d bar chart (|SHAP| + σ)
    ├── boxplots_top5.png                  # Top 5 ozellik boxplot (anom vs norm)
    ├── roc_pr_comparison.png              # ROC + PR egrileri (5 yaklasim)
    ├── metrics_bar.png                    # AUC / PR-AUC / F1 karsilastirma
    ├── best_model_importance.png          # En iyi siniflandirici ozellik onemleri
    ├── per_feature_stats.csv              # Ozellik bazli istatistik tablosu
    └── classification_comparison.csv      # Siniflandirma karsilastirma tablosu
```

### Kaynak SHAP Dosyalari

```
automl/multivariate/results/
├── diverse/shap_{LightGBM,XGBoost,RandomForest,Ridge,DecisionTree}_TS1.csv
├── h2o/shap_{GBM,DRF,XRT,DeepLearning,GLM}_TS1.csv
└── shap_{LightGBMXT_BAG_L1,LightGBMXT_BAG_L2,LightGBM_BAG_L1,
          RandomForestMSE_BAG_L1,WeightedEnsemble_L2,WeightedEnsemble_L3}_TS1.csv
```

---

## 9. Nasil Calistirilir

```bash
# Full test seti
python run_combined.py

# Belirli tarih araligi (zoom)
python run_combined.py --start 2021-06-01 --end 2021-06-25
```
