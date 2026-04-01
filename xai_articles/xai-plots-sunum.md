# Zaman Serisi Model Attribution Gorsellestirilmesi

**Kaynak:** Schlegel & Keim (2021) - "Time Series Model Attribution Visualizations as Explanations"

---

## 1. Giris ve Motivasyon

- Aciklanabilir Yapay Zeka (XAI), kara kutu modellerin kararlarini anlamak, hata ayiklamak ve iyilestirmek icin teknikler sunar
- Attribution (atif) yontemleri, tek bir ornek icin girdi degiskenlerinin onemini gosteren **yerel aciklama** teknikleridir
- Goruntulerde heatmap'ler iyi calisirken, **zaman serilerinde heatmap'ler yorumlamasi zordur**
- Saglik, ceza adaleti gibi kritik alanlarda aciklanabilirlik zorunludur

---

## 2. Zaman Serilerinde Attribution Yontemleri

### 2.1 Gradient Tabanli Yontemler
| Yontem | Aciklama |
|--------|----------|
| **Saliency** | Cikis noronlarindan girdi noronlarina gradient yayilimi |
| **GuidedBackpropagation** | Yonlendirilmis geri yayilim |
| **SmoothGrad** | Gurultuyu azaltmak icin girise gurultu ekleyerek gradyanlari yumusatir |
| **Integrated Gradients** | Bir baseline'dan girdiye dogru gradyanlarin integralini hesaplar |

- **Avantaj:** Hizli hesaplama
- **Dezavantaj:** Kirik gradyan (shattered gradients) problemi nedeniyle gurultulu olabilir

### 2.2 Yapi Tabanli Yontemler
| Yontem | Aciklama |
|--------|----------|
| **LRP** (Layer-wise Relevance Propagation) | Katman bazinda agirlik kurallariyla skor yayilimi |
| **DeepTaylor Decomposition** | Taylor acinimi ile skor hesaplama |
| **DeepLIFT** | Referans girdiye gore aktivasyon farklarini yayar |

- **Avantaj:** Gradyan sorunlarini asabilir, derin modellerde hizli
- **Dezavantaj:** Katman kurali secimi ve referans belirleme zorlugu

### 2.3 Surrogate / Ornekleme Yontemleri
| Yontem | Aciklama |
|--------|----------|
| **LIME** | Perturbasyonlu orneklerle yorumlanabilir model egitir |
| **SHAP** | Oyun teorisi tabanli aditif attribution skorlari |

- **Avantaj:** Model-agnostik, her modele uygulanabilir
- **Dezavantaj:** Ornekleme kalitesine bagimli, daha yavas

---

## 3. Heatmap Gorsellestirme Yaklasimlari

### 3.1 Cizgi Grafik Uzerinde Renk Kodlama
- **Van der Westhuizen & Lasenby:** LSTM icin ECG verisinde nokta boyutu + jet renk skalasi
  - Sorun: Overplotting, renk gradyaninda onemli noktalari bulmak zor

### 3.2 Cizgi Uzerinde Gradient Renklendirme
- **Siddiqui et al. (TSviz):** Cizginin kendisine renk gradyani uygulama
  - Sorun: Kotu gradyanlarda cizgi gorunurlugu azalabilir

### 3.3 Dikdortgen Heatmap (Dense-pixel)
- **Assaf & Schumann:** Zaman serisini kaldirip her zaman noktasini dikdortgen olarak gosterme (Grad-CAM)
  - Avantaj: Oruntu kesfine odaklanma
  - Sorun: Bazi oruntular uzman bilgisiyle bile aciklanmasi zor

### 3.4 Arka Plan Heatmap + Cizgi Grafik
- **Schlegel et al.:** Cizgi grafigin arkasina heatmap dikdortgenler yerlestirme (beyaz->kirmizi)
- **Jeyakumar et al.:** Benzer yaklasim, cyan->mor renk skalasi
  - Sorun: Uzman olmayanlar icin karmasik; yuksek relevans dusuk relevans yaninda gorunebilir

---

## 4. Alternatif Gorsellestirmeler: Cizgi Grafik Uzantilari

### 4.1 Ayri Attribution Cizgi Grafigi
- **Siddiqui et al. (TSinsight):** Attribution'i ayri bir cizgi grafik olarak gosterme
  - Avantaj: Veri ve attribution ayri incelenebilir
  - Dezavantaj: Iki grafik arasi iliskiyi kurmak zor

### 4.2 Boru (Pipe) Gorsellestirme
- **Mujkanovic et al. (timeXplain):** Cizgi etrafinda boru seklinde relevans gosterimi
  - Avantaj: Boyut + renk ile dikkat yonlendirme; dusuk relevans da gorunur kalir

### 4.3 Bar Grafik + Ok Karsilastirma
- **Xu et al. (MTSeer):** SHAP'in aditif ozelligini bar grafik olarak gosterme
  - Avantaj: Cok degiskenli serilerde ozellik etkilerini ve modeller arasi farklari karsilastirma
  - Dezavantaj: Tek degiskenli serilere uygulanamaz

---

## 5. Degerlendirme: Heatmap vs Ornekle Aciklama

- **Jeyakumar et al. kullanici calismasi sonucu:**
  - Ornekle aciklama (explanation by example) > Heatmap attribution (Grad-CAM++, Saliency, SHAP)
  - Ozellikle uzman olmayanlar icin heatmap'ler yetersiz

---

## 6. Gelecek Yonelimler ve Oneriler

### Onerilen Pipeline (Shneiderman Mantra Tabanli):

```
1. GENEL BAKIS  -->  Counterfactual aciklamalar
   (Tahmini degistiren minimum degisiklikleri goster)

2. YAKINLASTIR & FILTRELE  -->  Bireysel attribution gorsellestirmeleri
   (Veri alani duzeyinde detay)

3. TALEP UZERINE DETAY  -->  Attribution + zaman serisi etkilesimi
   (What-if analizi, zaman noktasi problama)
```

### Temel Argumanlar:
- **Uzman olmayanlar icin:** Counterfactual aciklamalar tercih edilmeli
- **Alan uzmanlari icin:** Attribution yontemleri detayli analiz icin kullanilmali
- **En iyi yaklasim:** Her ikisini birlestiren bir pipeline

---

## 7. Ozet Tablosu

| Gorsellestirme Turu | Avantaj | Dezavantaj | Hedef Kitle |
|---------------------|---------|------------|-------------|
| Nokta boyutu + renk | Basit, dogrudan | Overplotting | Uzman |
| Cizgi renk gradyani | Overplotting yok | Cizgi gorunurlugu dusuk | Uzman |
| Dense-pixel heatmap | Oruntu kesfine odakli | Veri kaybi | Uzman |
| Arka plan heatmap | Veri + attribution birlikte | Karmasik, yaniltici olabilir | Uzman |
| Ayri cizgi grafik | Net ayrim | Iliski kurmak zor | Uzman |
| Boru (pipe) | Boyut+renk ile sezgisel | Karmasik veriler icin olceklenme | Herkes |
| Bar grafik + oklar | Model karsilastirma | Tek degiskenli serilere uygulanamaz | Uzman |
| Counterfactual | Insan karar vermesine yakin | Uretimi zor olabilir | Herkes |

---

## 8. Anahtar Cikarimlar

1. Heatmap'ler goruntuler icin etkili ancak **zaman serileri icin yetersiz** kalabiliyor
2. Attribution teknigi secimi veri tipine ve model mimarisine gore **nicel degerlendirme** ile yapilmali
3. Tek bir gorsellestirme yeterli degil; **katmanli aciklama pipeline'i** (counterfactual + attribution) oneriliyor
4. Kullanici calismalari, **ornekle aciklamanin** heatmap attribution'dan daha etkili oldugunu gosteriyor
5. Gelecekte **etkilesimli gorsellestirmeler** ve **what-if analizleri** ile daha derin model incelemesi mumkun
