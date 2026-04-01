# Implet: Zaman Serisi Modelleri icin Post-hoc Alt-Dizi Aciklayici

**Kaynak:** Meng, Kan et al. (2025) - "Implet: A Post-hoc Subsequence Explainer for Time Series Models" - IEEE ICDMW 2025

---

## 1. Giris ve Motivasyon

- Derin ogrenme modelleri zaman serisi siniflandirmada basarili ancak **kara kutu** olarak kaliyor
- Mevcut aciklanabilirlik yontemleri (feature attribution) **tek tek zaman noktalarini** acikliyor -> yuksek boyutluluk, yorumu zorlastiriyor
- Shapelet tabanli yontemler daha sezgisel ama **inherent** (model egitimi sirasinda) ve SOTA mimarilerle uyumsuz
- **Implet** bu bosluyu dolduruyor: post-hoc + alt-dizi duzeyi + model-bagimli aciklama

---

## 2. Temel Kavramlar

### Mevcut XAI Yaklasimlari ve Sinirliliklari

| Yaklasim | Ornek | Avantaj | Dezavantaj |
|----------|-------|---------|------------|
| **Feature Attribution** | SHAP, LIME, Grad-CAM, Saliency | Post-hoc, model-agnostik | Yuksek boyutlu, "spike"li, yorumu zor |
| **Shapelet** | ShapeletTransform | Sezgisel alt-dizi aciklamalar | Inherent, SOTA modellerle uyumsuz |
| **Counterfactual** | Instance-based | Karar sinirlari gosterir | Hesaplama maliyeti yuksek |
| **LASTS** (tek post-hoc alt-dizi) | Decision tree + shapelet | Post-hoc | Cogu veri setinde anlamsiz sonuclar |

### Implet'in Farki
- **Post-hoc**: Herhangi bir egitilmis modele uygulanabilir
- **Alt-dizi duzeyi**: Tek nokta yerine anlamli temporal segmentler
- **Model-bilinir**: Attribution'dan turetildigi icin modelin gercek kararlarini yansitir

---

## 3. Implet Yontemi

### 3.1 Implet Cikarimi (Algorithm 1)

**Is Akisi:**
```
Zaman Serisi --> Model --> Attribution Hesapla --> Yuksek Attribution'lu
                                                    Ardisik Segmentleri Bul
                                                        --> IMPLET'ler
```

**Formal Tanim:**

Bir alt-dizi `I(l, r; x, w)` asagidaki kosulu sagliyorsa **Implet**'tir:

```
Skor: s(l, r) = (toplam |w_i|, i=l..r) + lambda * (r - l + 1) >= phi
Uzunluk: l_min <= r - l + 1 <= l_max
```

- **Ilk terim:** Alt-dizideki kumulatif attribution (yuksek olmalysa)
- **Ikinci terim (lambda):** Cok kisa segmentleri engeller (lambda = 0.1)
- **Esik (phi):** Normalizasyon sonrasi ~1 standart sapma uzerinde (phi = 1)
- **Uzunluk sinirlari:** l_min = 3, l_max = T/2

**Algoritma Ozeti:**
1. Ilk zaman adiminda attribution >= phi mi kontrol et
2. Evet ise, en iyi bitis noktasini bul (skoru maksimize eden)
3. Skor >= phi ise Implet olarak kaydet, aramaya bitis noktasindan devam et
4. Hayir ise bir sonraki zaman adimina gec
5. Karmasiklik: **O(T)** - tek bir ornek icin

---

### 3.2 Coh-Implet: Kohort Aciklamasi (Algorithm 2)

**Amac:** Benzer Implet'leri kumeleyerek daha **oz ve anlasilir** aciklamalar uretmek

**Global vs Lokal vs Kohort:**
| Seviye | Aciklama | Sorun |
|--------|----------|-------|
| Global | Tum model icin tek aciklama | Tum orneklere genellenmeyebilir |
| Lokal | Her ornek icin ayri aciklama | Cok fazla bilgi, gereksiz tekrar |
| **Kohort** | Benzer orneklerin grup aciklamasi | **Denge: oz + genellenebilir** |

**Kumeleme Detaylari:**
- Mesafe metrigi: **2-boyutlu bagimli DTW** (deger + attribution)
- Merkez hesaplama: **DTW Barycenter Averaging (DBA)**
- Optimal k secimi: **Silhouette skoru** ile otomatik
- DTW'nin secilme nedeni: Degisken uzunluklu Implet'lere uygunluk + pooling gibi ag operasyonlariyla uyum

---

## 4. Alt-Dizi Kaldirma Yontemi (Faithfulness Testi)

### Problem
Implet'in gercekten modelin karar verdigi bolgeleri bulup bulmadigini test etmek gerekiyor.

### Zorluk
- Sifir doldurma: Yapay sureksizlikler olusturur
- Kayan pencere ortalamasi: Duz alt-dizileri kaldirmada etkisiz
- Ornek ortalamasi: Ani gecisler olusturur

### Onerilen Cozum: Rasgelestrilmis Polinom Degistirme
1. Kontrol noktasi sayisi = max(ceil(L/10), 2)
2. Her kontrol noktasina ornekle ayni ortalama ve std'ye sahip rassal deger ata
3. Baslangic ve bitis degerleriyle interpolasyon yap + gradyan uyumu sagla

**Sonuc:** Yumusak gecisler, istenmeyen model tepkilerini minimize eder

---

## 5. Deneysel Sonuclar

### 5.1 Modeller ve Attribution Yontemleri

| Model | Ozellik |
|-------|---------|
| **FCN** (Fully Convolutional Network) | Basit, hizli |
| **InceptionTime** | SOTA, karmasik |

| Attribution | Tur |
|-------------|-----|
| Saliency | Gradient |
| Input x Gradient | Gradient |
| DeepLIFT | Yapi |
| GuidedBackprop | Gradient |
| LIME | Surrogate |
| KernelSHAP | Surrogate |
| Occlusion | Perturbation |

### 5.2 Nitel Analiz

**GunPoint Veri Seti (Ikili siniflandirma: silah cekme vs parmakla isaret):**
- Her sinif 2 kume olusturdu: yukari hareket + asagi hareket
- Fiziksel sezgiyle uyumlu (silah vs parmak farkli ivme profilleri)
- Farkli deneklerden (erkek/kadin) gelen benzer alt-diziler ayni kumede

**Chinatown Veri Seti (Hafta ici vs hafta sonu yaya trafigi):**
- Her sinif 1 ana kume: sabah erken saatler (01:00-06:00) onemli
- Sezgiyle uyumlu: hafta sonu gece hayati vs hafta ici erken uyku

### 5.3 Nicel Analiz (Faithfulness)

**Test Yontemi:** Implet kaldir -> dogruluk dususu olcumu vs rassal alt-dizi kaldirma

**Temel Bulgular:**
- Implet kaldirmak **her zaman** rassal kaldirmadan daha buyuk dogruluk dususune neden oluyor
- En iyi attribution yontemleri: **Saliency, Input x Gradient, DeepLIFT**
- LIME ve KernelSHAP seyrek/parcali attribution urettigi icin Implet performansi dusuk
- GuidedBackprop basit modellerde (FCN) iyi, derin modellerde (InceptionTime) zayif
- ShapeletTransform: uzun ve gurultulu alt-diziler -> rassal kaldirma da benzer etki yapiyor

### 5.4 Coh-Implet Faithfulness

- Coh-Implet merkezlerine benzer alt-diziler (CILS) bulunup kaldirildi
- **Attribution bilgisi olmadan bile** benzer dogruluk dusus orani elde edildi
- Siralam: 1D CILS < 2D CILS < Implet (beklenen sonuc)
- Sonuc: Kumeleme sonrasi merkezler modele sadik kaliyor

---

## 6. Sinirliliklar

| Durum | Sorun | Cozum |
|-------|-------|-------|
| **Kisa seriler** (Chinatown, 24 adim) | Cok kisa Implet'ler, yumusak kaldirma etkisiz | Ortalama doldurma ile daha iyi sonuc |
| **Frekans tabanli** (FordA) | Alt-dizi aciklamalari yetersiz | Birden fazla Implet'i birlikte kaldirmak |
| **Olay tabanli** (Earthquakes) | Tek alt-dizi kaldirma yeterli degil | Tum Implet'leri birlikte kaldirmak |
| **Cok degiskenli seriler** | Henuz test edilmedi | 2n-boyutlu Implet genellemesi oneriliyor |

---

## 7. Model Performanslari (UCR Veri Setleri)

| Veri Seti | FCN | InceptionTime |
|-----------|-----|---------------|
| GunPoint | %100 | %98.67 |
| Chinatown | %98.54 | %97.67 |
| Coffee | %96.43 | %96.43 |
| ECG200 | %86.00 | %89.00 |
| ECGFiveDays | %95.94 | %99.77 |
| Strawberry | %97.30 | %100 |
| FordA | %90.91 | %94.24 |

---

## 8. Ozet ve Karsilastirma Tablosu

| Ozellik | Feature Attribution | Shapelet | LASTS | **Implet** |
|---------|-------------------|----------|-------|-----------|
| Post-hoc | Evet | Hayir | Evet | **Evet** |
| Alt-dizi duzeyi | Hayir | Evet | Evet | **Evet** |
| Model-bilinir | Evet | Hayir | Kismi | **Evet** |
| Kohort aciklama | Hayir | Hayir | Hayir | **Evet (Coh-Implet)** |
| SOTA model uyumu | Evet | Hayir | Sinirli | **Evet** |
| Hesaplama | Degisken | Yuksek | Yuksek | **O(T)** |

---

## 9. Anahtar Cikarimlar

1. **Implet**, feature attribution'in **post-hoc avantajini** shapelet'in **alt-dizi sezgiselligiyle** birlestiren ilk yontemdir
2. **Coh-Implet**, zaman serilerinde ilk kohort aciklama cercevesidir - gereksiz tekrari azaltir, oz aciklamalar sunar
3. En iyi attribution eslesmesi: **Saliency ve Input x Gradient** - tutarli yuksek kaliteli Implet'ler uretir
4. **Rasgelestrilmis polinom** kaldirma yontemi, zaman serilerinde ablasyon analizinde yapay etkileri minimize eder
5. Kumeleme sonrasi **Coh-Implet merkezleri** attribution bilgisi olmadan bile modele sadik kaliyor
6. Frekans/olay tabanli verilerde alt-dizi aciklamalari sinirli kaliyor - gelecekte alan-ozel cozumler gerekiyor
