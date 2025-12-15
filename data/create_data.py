# data/create_data.py - GÜNCELLENMİŞ BÜYÜK VERİ SETİ
import pandas as pd
import random

print("📊 BÜYÜK VERİ SETİ OLUŞTURULUYOR (200 e-posta)...")

# Daha zengin spam/ham örnekleri
spam_examples = [
    # Finansal spam
    "WIN $1,000,000 NOW! Click the link to claim your prize immediately!",
    "URGENT: Your bank account security has been compromised. Verify now!",
    "Earn $5000 monthly working from home, no experience required at all!",
    "Limited time offer: Get rich quick with our exclusive investment program!",
    "Your credit card has unusual activity. Confirm your identity now!",
    
    # Ürün spam
    "Lose 10kg in 7 days with this miracle diet pill. Guaranteed results!",
    "Special 90% discount: Viagra, Cialis, weight loss pills available now!",
    "Free iPhone 15 for you! Just complete our short survey to claim!",
    "Your Netflix subscription is about to expire. Update payment now!",
    "Exclusive offer: Luxury watch originally $1000, now only $99!",
    
    # Acil spam
    "FINAL NOTICE: Your account will be suspended in 24 hours!",
    "IMPORTANT: You have unclaimed tax refund of $852. Click here!",
    "Your PayPal account needs immediate verification to continue!",
    "Security alert: Someone tried to access your email from new device!",
    "You won a free vacation to Hawaii! Claim before it's gone!",
    
    # Tekrar varyasyonları
    "CONGRATULATIONS! You are selected for $50,000 cash prize!",
    "Make money fast with our proven system. Start earning today!",
    "Your Amazon Prime is expiring. Renew now to keep benefits!",
    "Act now! Last chance to get iPhone at 80% discount!",
    "Warning: Your social security number may be at risk!"
]

ham_examples = [
    # İş e-postaları
    "Hi team, the quarterly financial report is attached for your review.",
    "Meeting scheduled for tomorrow at 2 PM in the main conference room.",
    "Please find attached the project timeline and deliverables document.",
    "Reminder: Department all-hands meeting this Friday at 10 AM.",
    "The software update has been successfully deployed to production.",
    
    # Kişisel e-postalar
    "Hi John, can you send me the sales figures for last quarter?",
    "Your Amazon package #12345 has been delivered to your doorstep.",
    "Monthly team lunch will be this Thursday at the Italian restaurant.",
    "Please review and sign the attached contract by end of day.",
    "Server maintenance is scheduled for Saturday from 2 AM to 4 AM.",
    
    # Proje yönetimi
    "Project deadline has been extended to next Friday as requested.",
    "New company policy updates are available on the HR portal.",
    "Weekly status report: All tasks are on track for completion.",
    "The marketing campaign results exceeded our expectations.",
    "IT announcement: New VPN configuration guidelines released.",
    
    # Tekrar varyasyonları
    "Can we schedule a call tomorrow to discuss the proposal?",
    "Your expense report for March has been approved and processed.",
    "Team building event planned for next month. Please RSVP.",
    "The quarterly review meeting notes are available on SharePoint.",
    "New feature deployment scheduled for next Tuesday morning."
]

# 100 spam + 100 ham = 200 e-posta
all_emails = []
all_labels = []

print("Generating 100 spam emails...")
for _ in range(100):
    email = random.choice(spam_examples)
    # Varyasyon ekle
    if random.random() > 0.5:
        email = email.replace("$1,000,000", f"${random.randint(1000, 10000)}")
    all_emails.append(email)
    all_labels.append(1)

print("Generating 100 ham emails...")
for _ in range(100):
    email = random.choice(ham_examples)
    # Varyasyon ekle
    if random.random() > 0.5:
        email = email.replace("tomorrow", random.choice(["Monday", "Tuesday", "Wednesday"]))
    all_emails.append(email)
    all_labels.append(0)

# DataFrame oluştur
data = {
    'text': all_emails,
    'label': all_labels
}

df = pd.DataFrame(data)

# Karıştır
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Kaydet
df.to_csv('data/emails.csv', index=False)

print(f"\n✅ BÜYÜK VERİ SETİ OLUŞTURULDU!")
print(f"   Toplam: {len(df)} e-posta")
print(f"   Spam: {sum(df['label'])}")
print(f"   Ham: {len(df)-sum(df['label'])}")
print(f"   Spam oranı: {sum(df['label'])/len(df):.1%}")
print(f"   Dosya: data/emails.csv")

# İstatistikler
print("\n📊 İSTATİSTİKLER:")
print(f"   Ortalama karakter: {df['text'].apply(len).mean():.0f}")
print(f"   Ortalama kelime: {df['text'].apply(lambda x: len(x.split())).mean():.0f}")