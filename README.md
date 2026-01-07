# Türkiye Elektrik Piyasası Fiyat Tahmini (2013-2016)

Bu proje, 2013-2016 yılları arasındaki Türkiye Gün Öncesi Piyasası (PTF) verilerini kullanarak elektrik fiyat tahmini yapmaktadır. Proje kapsamında makine öğrenmesi ve derin öğrenme yöntemleri karşılaştırmalı olarak analiz edilmiştir.

## Proje Özeti
- **Veri Seti:** 2013-2016 Türkiye Elektrik Fiyatları
- **Eğitim Seti (Train):** 2013 - 2014 yılları
- **Doğrulama Seti (Validation):** 2015 yılı
- **Test Seti (Test):** 2016 yılı

## Uygulanan Yöntemler
Proje yönergesine uygun olarak aşağıdaki modeller eğitilmiş ve performansları kıyaslanmıştır:
1. **LASSO:** İstatistiksel doğrusal model.
2. **XGBoost:** Karar ağacı tabanlı gelişmiş makine öğrenmesi algoritması.
3. **DNN (Deep Neural Network):** Çok katmanlı yapay sinir ağları.
4. **RNN (LSTM):** Zaman serisi verilerindeki ardışık bağımlılıkları yakalamak için geliştirilen uzun-kısa süreli bellek modeli.

## Özellik Mühendisliği (Feature Engineering)
Yönergede belirtilen Medium makaleleri doğrultusunda, verinin zaman serisi karakteristiğini güçlendirmek için şu "Lag" (Gecikmeli) özellikler eklenmiştir:
- **Price_Lag24:** 24 saat öncesinin fiyatı.
- **Price_Lag168:** 1 hafta öncesinin (168 saat) fiyatı.

## Sonuçlar (MAE)
Modellerin 2016 yılı test verisi üzerindeki Ortalama Mutlak Hata (MAE) sonuçları:
- **DNN:** 24.44 TL
- **LSTM:** 25.71 TL
- **XGBoost:** 25.84 TL
- **LASSO:** 26.13 TL
