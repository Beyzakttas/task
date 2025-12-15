# 🚨 Spam E-Posta Tespit Sistemi

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Doğal Dil İşleme (NLP) ve Makine Öğrenmesi kullanarak spam e-postaları otomatik tespit eden bir sınıflandırma sistemi.**

---

## 📋 İçindekiler
- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Teknolojiler](#-teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Performansı](#-model-performansı)
- [Proje Yapısı](#-proje-yapısı)
- [Görseller](#-görseller)
- [Sonuçlar](#-sonuçlar)
- [Geliştirici](#-geliştirici)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

Bu proje, bir şirkete gelen binlerce e-postanın otomatik olarak spam/ham (istenmeyen/normal) olarak sınıflandırılması amacıyla geliştirilmiştir. Proje kapsamında:

- **200 örnek e-posta** içeren bir veri seti oluşturuldu
- **4 farklı makine öğrenmesi modeli** karşılaştırıldı
- **NLP pipeline'ı** kuruldu (ön işleme, vektörleme, modelleme)
- **%94.5 accuracy** elde edildi
- **Gerçek zamanlı tahmin fonksiyonu** geliştirildi

---

## ✨ Özellikler

- ✅ **Metin Ön İşleme**: Küçük harfe çevirme, noktalama temizleme, stopwords kaldırma, lemmatization
- ✅ **Feature Engineering**: TF-IDF vektörleme (1500 özellik, bigram desteği)
- ✅ **Çoklu Model Testi**: Naive Bayes, Logistic Regression, SVM, Random Forest
- ✅ **Detaylı Değerlendirme**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ **Cross-Validation**: 5-fold cross validation ile model stabilitesi testi
- ✅ **Tahmin Fonksiyonu**: Gerçek zamanlı e-posta sınıflandırma
- ✅ **Zengin Görseller**: 7+ grafik ile detaylı analiz

---

## 🛠️ Teknolojiler

| Teknoloji | Açıklama | Versiyon |
|-----------|----------|----------|
| **Python** | Programlama dili | 3.9+ |
| **Pandas** | Veri işleme | 2.0+ |
| **NumPy** | Bilimsel hesaplamalar | 1.24+ |
| **Scikit-learn** | Makine öğrenmesi | 1.3.0 |
| **NLTK** | Doğal Dil İşleme | 3.8+ |
| **Matplotlib** | Görselleştirme | 3.7+ |
| **Seaborn** | İstatistiksel görselleştirme | 0.12+ |
