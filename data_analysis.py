# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Excel dosyasını oku
df = pd.read_excel('data/chatbot_dataset.xlsx')

# Sütun isimlerini kontrol et
print("Sütunlar:", df.columns.tolist())
print("\nVeri şekli:", df.shape)

# Sütun isimlerini standartlaştır: category -> intent, sentence -> text
df = df.rename(columns={'category': 'intent', 'sentence': 'text'})

print("\nYeni sütun isimleri:", df.columns.tolist())
print("\nIntent dağılımı:")
print(df['intent'].value_counts())

# Intent dağılımını görselleştir
plt.figure(figsize=(12, 6))
intent_counts = df['intent'].value_counts()

# Bar plot
plt.subplot(1, 2, 1)
intent_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Intent Dağılımı', fontsize=14, fontweight='bold')
plt.xlabel('Intent', fontsize=12)
plt.ylabel('Örnek Sayısı', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Değerleri bar üzerine ekle
for i, v in enumerate(intent_counts.values):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold')

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%',
        startangle=90, colors=plt.cm.Set3.colors)
plt.title('Intent Oranları', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('intent_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Görsel kaydedildi: intent_distribution.png")

# Temel istatistikler
print("\n" + "="*50)
print("VERİ SETİ İSTATİSTİKLERİ")
print("="*50)
print(f"Toplam örnek sayısı: {len(df)}")
print(f"Intent sayısı: {df['intent'].nunique()}")
print(f"\nIntent'ler: {df['intent'].unique().tolist()}")

# Ortalama cümle uzunluğu
df['word_count'] = df['text'].str.split().str.len()
print(f"\nOrtalama kelime sayısı: {df['word_count'].mean():.2f}")
print(f"Min kelime sayısı: {df['word_count'].min()}")
print(f"Max kelime sayısı: {df['word_count'].max()}")

# Her intent için ortalama uzunluk
print("\nIntent'lere göre ortalama kelime sayısı:")
print(df.groupby('intent')['word_count'].mean().round(2))

# Boş değer kontrolü
print("\n" + "="*50)
print("VERİ KALİTESİ KONTROLÜ")
print("="*50)
print(f"Boş değer sayısı:")
print(df.isnull().sum())

# Duplicate kontrolü
duplicates = df.duplicated(subset=['text']).sum()
print(f"\nTekrarlayan cümle sayısı: {duplicates}")

if duplicates > 0:
    print(f"  → {duplicates} duplicate bulundu, temizlenecek.")

# Standartlaştırılmış versiyonu kaydet
df_clean = df[['intent', 'text']].copy()  # Sadece gerekli sütunlar
df_clean.to_excel('data/chatbot_dataset_standardized.xlsx', index=False)
print("\n✓ Standartlaştırılmış veri kaydedildi: data/chatbot_dataset_standardized.xlsx")

plt.show()