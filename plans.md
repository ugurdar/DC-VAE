# 21.04.2026

ts1'in anomalileri için ts2-ts12 değerlerini kullanıp sınıflandırma yapmak.

https://www.colibri.udelar.edu.uy/jspui/bitstream/20.500.12008/41900/1/GMFGAC23.pdf


1- değişkenlerin etkileri zaman göre nasıl değişiyor, son sunduğumzu analizi tamamlamak, hem rashomon için threshold kullanmak, forecast bunun bir kısmını yaptık, en yüksek performanslı modele göre, yüksek performans gösteren modelleri ekle. bir de daha dar pencere de dene. değişimleri görmek için. burada ts2-ts12 predictor olarak kullanılacak.

2- açıklamalar üzerine anomalileri koyup, açıklamaların belirsizliğinin artmasıyla anomaliler arasında korelasyon var mı, korelasyon katsayısını araştır kategorik vs numerik. point-biserial. model de dene.
1.de yaptığımızı anomliye, her değişkenden gelen açıklama belirsizlğini anomaliyi tahmin etmek için.
açıklama belirsizliği vs anomali.

3- tüm deney kurgusunu time series clasification anomailileri.
açıklama belirsizliği tespit etmek yerine anomalileri sınıflandıran modeli rashomon ile açıklama
ts1_label ~ ts_2 + ... + ts_12 classification

x ekseni zaman, x eksenini bir de o değişkenin değerleri olarak dene


4- bunlar haricinde zamandan bağımsız bu değerlerin bir katkısı var mı. öneri olarak sunabiliriz.

5- (ileride) sentetik.
---

## TELCO Veri Setinde Anomali Tespiti: Literatür Taraması

### 1. TELCO Veri Seti Hakkında

TELCO veri seti, Uruguay'daki operasyonel bir mobil ISP'den (Telefónica Uruguay) toplanan, **12 çok değişkenli zaman serisi** içeren bir ağ izleme veri setidir. 5 dakika granülariteyle, Ocak-Temmuz 2021 arasında 7 ay boyunca toplanmış ve uzmanlar tarafından **manuel olarak etiketlenmiştir**. Veri seti IEEE DataPort'ta yayınlanmıştır (DOI: 10.21227/skpg-0539).

Yaratıcıları: García González, Martínez Tagliafico, Fernández, Gómez, Acuña ve **Pedro Casas** (AIT Austrian Institute of Technology).

### 2. TELCO Veri Setini Kullanan Makaleler ve Sonuçları

#### 2.1 DC-VAE (IEEE EuroS&PW 2022 + IEEE TNSM 2023)
- **Makale**: "DC-VAE, Fine-grained Anomaly Detection in MTS with Dilated Convolutions and VAEs" + "One Model to Find Them All"
- **Link**: https://ieeexplore.ieee.org/document/9799327/ ve https://ieeexplore.ieee.org/document/10345720/
- **Sonuç**: 12 zaman serisinden **8'inde F1 ~%60**, kalan 4'ünde daha düşük
- **Not**: TELCO veri setini oluşturan ve ilk kullanan çalışma

#### 2.2 Net-GAN (arXiv 2020)
- **Makale**: "On the Usage of Generative Models for Network Anomaly Detection in Multivariate Time-Series"
- **Link**: https://arxiv.org/abs/2010.08286
- **Yazarlar**: García González, Casas, Fernández, Gómez
- **Yöntem**: RNN + GAN tabanlı (Net-GAN ve Net-VAE)
- **Not**: DC-VAE'nin öncüsü, aynı ekip tarafından, ISP ağ verisi üzerinde değerlendirilmiş

#### 2.3 GNNs for TSAD (arXiv 2025)
- **Makale**: "GNNs for Time Series Anomaly Detection: An Open-Source Framework and a Critical Evaluation"
- **Link**: https://arxiv.org/abs/2603.09675
- **Yazarlar**: Bello, Chiarlone, Fiori, García González, Larroca
- **Veri Setleri**: TELCO + SWaT
- **TELCO Sonuçları**:

| Model | Precision | Recall | F1 | P_T | R_T | F1_T | VUS-ROC | VUS-PR |
|-------|-----------|--------|----|-----|-----|------|---------|--------|
| GCN | 0.15 | 0.10 | 0.08 | 0.09 | 0.29 | 0.11 | 0.64 | 0.05 |
| GDN | 0.32 | 0.18 | 0.11 | 0.30 | 0.48 | 0.25 | 0.62 | 0.08 |
| MTAD-GAT | 0.39 | 0.16 | 0.10 | 0.34 | 0.44 | 0.25 | 0.61 | 0.07 |
| GRU | 0.39 | 0.14 | 0.12 | 0.35 | 0.48 | 0.30 | 0.58 | 0.09 |

- **Aynı modellerin SWaT sonuçları** (karşılaştırma için):

| Model | Precision | Recall | F1 | P_T | R_T | F1_T | VUS-ROC | VUS-PR |
|-------|-----------|--------|----|-----|-----|------|---------|--------|
| GCN | 0.80 | 0.79 | 0.80 | 0.06 | 0.33 | 0.10 | 0.72 | 0.55 |
| GDN | 1.00 | 0.75 | 0.85 | 1.00 | 0.07 | 0.13 | 0.85 | 0.73 |
| MTAD-GAT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.88 | 0.74 |
| GRU | 0.98 | 0.76 | 0.86 | 0.08 | 0.24 | 0.12 | 0.86 | 0.77 |

#### 2.4 Foundation Auto-Encoders / FAE (arXiv 2025)
- **Makale**: "Towards Foundation Auto-Encoders for Time-Series Anomaly Detection"
- **Link**: https://arxiv.org/abs/2507.01875
- **Yazarlar**: García González, Casas, Martínez, Fernández
- **Veri Setleri**: TELCO + TELCO2 (2024 verisi) + KDD2021
- **Sonuç**: DC-VAE ile karşılaştırılabilir performans, nitel analiz (sayısal F1 raporlanmamış)
- **Not**: VAE tabanlı foundation model yaklaşımı, ön sonuçlar

### 3. Önemli Çıkarımlar

1. **TELCO çok zor bir veri seti**: GNN modelleri (GCN, GDN, MTAD-GAT) TELCO'da F1 = 0.08-0.12 alırken, aynı modeller SWaT'ta F1 = 0.80-0.86 alıyor
2. **DC-VAE hâlâ TELCO'daki en iyi sonuç**: 8/12 seride F1 ~%60, diğer yöntemlerden çok daha iyi
3. **TELCO'yu kullanan tüm makaleler aynı ekipten**: García González, Casas ve arkadaşları (Uruguay + Avusturya)
4. **Veri seti henüz yaygın kullanılmıyor**: 2023'te yayınlanmasına rağmen dış gruplardan kullanım sınırlı
5. **Point-adjust metrikleri (P_T, R_T, F1_T) bile düşük**: TELCO'nun gerçek dünya zorluğunu gösteriyor

### Kaynaklar (Doğrudan TELCO Kullananlar)

- [DC-VAE (IEEE EuroS&PW 2022)](https://ieeexplore.ieee.org/document/9799327/)
- [One Model to Find Them All (IEEE TNSM 2023)](https://ieeexplore.ieee.org/document/10345720/)
- [Net-GAN (arXiv 2020)](https://arxiv.org/abs/2010.08286)
- [GNNs for TSAD (arXiv 2025)](https://arxiv.org/abs/2603.09675)
- [Foundation Auto-Encoders / FAE (arXiv 2025)](https://arxiv.org/abs/2507.01875)
- [TELCO Dataset (IEEE DataPort)](https://ieee-dataport.org/documents/telco)

### Kaynaklar (Genel Telekom/MTS Anomali Tespiti Referansları)

- [AI Advances in Telecom Anomaly Detection (Springer 2025)](https://link.springer.com/article/10.1007/s10462-025-11108-x)
- [MTS Anomaly Detection Survey (Sensors 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/)
- [Deep Learning for TSAD Survey (ACM 2024)](https://dl.acm.org/doi/10.1145/3691338)
- [Reconstruction-based Methods Evaluation (Springer 2025)](https://link.springer.com/article/10.1007/s10462-025-11401-9)
- [GAT-Informer Benchmark on SWaT/WADI (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10935277/)
- [CESNET-TimeSeries24 ISP Dataset (Nature 2025)](https://www.nature.com/articles/s41597-025-04603-x)