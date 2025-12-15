# main.py - GÜNCELLENMİŞ (daha iyi performans)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# src klasörünü Python path'ine ekle
sys.path.append('src')

from src.preprocess import TextPreprocessor
from src.model import SpamClassifier

def main():
    print("=" * 70)
    print("SPAM E-POSTA TESPİT SİSTEMİ - GÜNCELLENMİŞ VERSİYON")
    print("=" * 70)
    
    # 1. VERİ YÜKLEME
    print("\n1. 📥 VERİ YÜKLENİYOR...")
    try:
        df = pd.read_csv('data/emails.csv')
        print(f"   ✅ {len(df)} e-posta yüklendi")
        print(f"   ✅ Spam: {sum(df['label'])}, Ham: {len(df)-sum(df['label'])}")
        print(f"   ✅ Spam oranı: {sum(df['label'])/len(df):.1%}")
    except Exception as e:
        print(f"   ❌ Hata: {e}")
        return
    
    # 2. VERİ ÖN İŞLEME
    print("\n2. 🧹 VERİ ÖN İŞLEME...")
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df)
    
    # Kelime sayısı analizi
    df_clean['word_count'] = df_clean['cleaned_text'].apply(lambda x: len(x.split()))
    print(f"   ✅ Ortalama kelime (temizlenmiş): {df_clean['word_count'].mean():.1f}")
    
    # 3. FEATURE ENGINEERING (DAHA İYİ TF-IDF)
    print("\n3. 🔧 FEATURE ENGINEERING (Gelişmiş TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=1500, 
        stop_words='english',
        ngram_range=(1, 2),  # Bigram ekle
        min_df=2,            # En az 2 dokümanda geçsin
        max_df=0.85          # %85'ten fazla dokümanda geçmesin
    )
    X = vectorizer.fit_transform(df_clean['cleaned_text'])
    y = df_clean['label']
    
    print(f"   ✅ Özellik matrisi: {X.shape}")
    print(f"   ✅ Toplam kelime/bigram: {len(vectorizer.get_feature_names_out())}")
    
    # En önemli 10 feature
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = np.asarray(X.mean(axis=0)).ravel()
    top_indices = feature_importance.argsort()[-10:][::-1]
    print(f"   ✅ En önemli kelimeler: {feature_names[top_indices][:5]}")
    
    # 4. TRAIN-TEST SPLIT
    print("\n4. ✂️ TRAIN-TEST AYRIMI...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   ✅ Eğitim seti: {X_train.shape[0]} örnek")
    print(f"   ✅ Test seti: {X_test.shape[0]} örnek")
    print(f"   ✅ Eğitim Spam oranı: {y_train.mean():.1%}")
    print(f"   ✅ Test Spam oranı: {y_test.mean():.1%}")
    
    # 5. MODEL EĞİTİMİ (4 MODEL)
    print("\n5. 🤖 MODEL EĞİTİMİ (4 Model)...")
    
    models_to_train = [
        ('Naive Bayes', 'naive_bayes'),
        ('Logistic Regression', 'logistic'),
        ('SVM', 'svm'),
        ('Random Forest', 'random_forest')
    ]
    
    results = []
    
    for name, model_type in models_to_train:
        print(f"   🔄 {name} eğitiliyor...")
        classifier = SpamClassifier(model_type)
        classifier.train(X_train, y_train)
        
        # Test tahmini
        y_pred = classifier.predict(X_test)
        
        # Cross-validation (5-fold)
        cv_scores = cross_val_score(classifier.model, X_train, y_train, cv=5, scoring='accuracy')
        
        metrics = {
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        results.append(metrics)
        
        print(f"     ✓ Test Accuracy: {metrics['accuracy']:.2%}")
        print(f"     ✓ CV Accuracy: {metrics['cv_mean']:.2%} (±{metrics['cv_std']:.2%})")
    
    # 6. SONUÇLARI GÖSTER
    print("\n6. 📊 PERFORMANS KARŞILAŞTIRMASI:")
    print("   " + "-" * 85)
    print("   {:<20} {:<10} {:<10} {:<10} {:<10} {:<12} {:<10}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'CV Mean', 'CV Std'))
    print("   " + "-" * 85)
    
    for res in results:
        print("   {:<20} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%} {:<12.2%} {:<10.2%}".format(
            res['model'],
            res['accuracy'],
            res['precision'],
            res['recall'],
            res['f1'],
            res['cv_mean'],
            res['cv_std']
        ))
    
    # 7. EN İYİ MODEL
    print("\n7. 🏆 EN İYİ MODEL SEÇİMİ:")
    best_model = max(results, key=lambda x: x['accuracy'])
    print(f"   ✓ En iyi model: {best_model['model']}")
    print(f"   ✓ Test Accuracy: {best_model['accuracy']:.2%}")
    print(f"   ✓ Precision: {best_model['precision']:.2%}")
    print(f"   ✓ Recall: {best_model['recall']:.2%}")
    print(f"   ✓ F1-Score: {best_model['f1']:.2%}")
    print(f"   ✓ CV Stability: {best_model['cv_mean']:.2%} (±{best_model['cv_std']:.2%})")
    
    # 8. DETAYLI CLASSIFICATION REPORT
    print("\n8. 📋 DETAYLI PERFORMANS ANALİZİ:")
    best_classifier = None
    for name, model_type in models_to_train:
        if name == best_model['model']:
            best_classifier = SpamClassifier(model_type)
            best_classifier.train(X_train, y_train)
            y_pred_best = best_classifier.predict(X_test)
            
            print(f"\n   {name} - Classification Report:")
            print("   " + "-" * 50)
            report = classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam'])
            # Satır satır yazdır
            for line in report.split('\n'):
                print(f"   {line}")
    
    # 9. CONFUSION MATRIX ÇİZ
    print("\n9. 📊 CONFUSION MATRIX GÖRSELLEŞTİRME...")
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'{best_model["model"]} - Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    
    plt.subplot(1, 2, 2)
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'{best_model["model"]} - Normalized Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    
    plt.tight_layout()
    plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✅ improved_confusion_matrix.png kaydedildi")
    
    # 10. TAHMİN FONKSİYONU
    print("\n10. 🔮 TAHMİN FONKSİYONU TESTİ:")
    
    # Tüm veriyle yeniden eğit (üretime hazır)
    best_classifier.train(X, y)
    
    def predict_spam(email_text):
        """Geliştirilmiş tahmin fonksiyonu"""
        cleaned = preprocessor.clean_text(email_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = best_classifier.predict(vectorized)[0]
        proba = best_classifier.predict_proba(vectorized)[0]
        
        result = 'SPAM' if prediction == 1 else 'HAM'
        confidence = max(proba) * 100
        
        return result, proba[1] * 100, confidence
    
    # Test örnekleri
    test_emails = [
        'Win a free iPhone 15 now! Click here to claim immediately!',
        'Team meeting scheduled for Monday at 10 AM in conference room B',
        'URGENT: Your bank account needs immediate verification!',
        'Please review the attached quarterly financial report',
        'Earn $10,000 monthly from home with no experience required',
        'Your package has been delivered to the front door',
        'Limited time offer: Get rich with our investment program',
        'Hi John, can you send me the project timeline by EOD?',
        'Warning: Your Netflix subscription is expiring tomorrow!',
        'The server maintenance is complete and all systems are operational'
    ]
    
    print("\n   📧 GERÇEKÇİ TEST SONUÇLARI:")
    print("   " + "=" * 70)
    print("   {:<50} {:<10} {:<12} {:<10}".format(
        'E-posta', 'Tahmin', 'Spam Olasılığı', 'Güven'))
    print("   " + "=" * 70)
    
    for email in test_emails:
        result, spam_prob, confidence = predict_spam(email)
        
        # Kısa gösterim
        short_email = (email[:47] + '...') if len(email) > 50 else email
        
        print("   {:<50} {:<10} {:<12.1f}% {:<10.1f}%".format(
            short_email, result, spam_prob, confidence
        ))
    
    # 11. PROJE ÖZETİ
    print("\n" + "=" * 70)
    print("📋 PROJE ÖZETİ (RAPOR İÇİN):")
    print("=" * 70)
    print(f"   • Veri seti: {len(df)} e-posta")
    print(f"   • Spam/Ham: {sum(df['label'])}/{len(df)-sum(df['label'])} ({sum(df['label'])/len(df):.1%})")
    print(f"   • Özellik sayısı: {X.shape[1]}")
    print(f"   • Test edilen modeller: {len(models_to_train)}")
    print(f"   • En iyi model: {best_model['model']}")
    print(f"   • Test Accuracy: {best_model['accuracy']:.2%}")
    print(f"   • Test Precision: {best_model['precision']:.2%}")
    print(f"   • Test Recall: {best_model['recall']:.2%}")
    print(f"   • Test F1-Score: {best_model['f1']:.2%}")
    print(f"   • Cross-Validation: {best_model['cv_mean']:.2%} (±{best_model['cv_std']:.2%})")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ GÜNCELLENMİŞ PROJE BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)
    
    # Ek bilgiler
    print("\n💡 RAPOR İÇİN EK BİLGİLER:")
    print(f"   • Toplam e-posta: {len(df)}")
    print(f"   • Eğitim örnekleri: {X_train.shape[0]}")
    print(f"   • Test örnekleri: {X_test.shape[0]}")
    print(f"   • TF-IDF ngram_range: (1, 2)")
    print(f"   • Feature sayısı: {X.shape[1]}")
    print(f"   • Cross-validation folds: 5")

if __name__ == '__main__':
    main()