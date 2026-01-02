# data_preprocessing.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Metni temizle"""
    # String kontrolü
    if not isinstance(text, str):
        text = str(text)
    
    # Küçük harfe çevir
    text = text.lower()
    
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_dataset(filepath):
    """Veri setini hazırla"""
    # Veriyi oku
    df = pd.read_excel(filepath)
    
    print(f"Orijinal sütunlar: {df.columns.tolist()}")
    
    # Sütun isimlerini standartlaştır
    # category -> intent, sentence -> text
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'category' in col_lower or 'intent' in col_lower:
            column_mapping[col] = 'intent'
        elif 'sentence' in col_lower or 'text' in col_lower:
            column_mapping[col] = 'text'
    
    df = df.rename(columns=column_mapping)
    
    print(f"Yeni sütunlar: {df.columns.tolist()}")
    
    # Gerekli sütunlar var mı kontrol et
    if 'intent' not in df.columns or 'text' not in df.columns:
        raise ValueError("Excel'de 'intent' veya 'text' sütunu bulunamadı!")
    
    # Sadece gerekli sütunları al
    df = df[['intent', 'text']].copy()
    
    # Boş değerleri temizle
    initial_count = len(df)
    df = df.dropna()
    print(f"Boş değer temizlendi: {initial_count} -> {len(df)} satır")
    
    # Metinleri temizle
    df['text'] = df['text'].apply(clean_text)
    df['intent'] = df['intent'].apply(lambda x: x.lower().strip())
    
    # Duplicate kontrolü
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text'])
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"Duplicate temizlendi: {duplicates_removed} satır")
    
    print(f"\n✓ Temizlenmiş veri: {len(df)} satır")
    print(f"✓ Intent sayısı: {df['intent'].nunique()}")
    print(f"✓ Intent'ler: {df['intent'].unique().tolist()}")
    
    # Intent dağılımını göster
    print("\nIntent Dağılımı:")
    print(df['intent'].value_counts())
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Veriyi train-test olarak böl"""
    
    print(f"\nVeri bölünüyor (Train: %{int((1-test_size)*100)}, Test: %{int(test_size*100)})...")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['intent']  # Her intent'ten dengeli dağılım
    )
    
    print(f"✓ Train: {len(train_df)} satır")
    print(f"✓ Test: {len(test_df)} satır")
    
    # Train-test dağılımını kontrol et
    print("\nTrain Intent Dağılımı:")
    print(train_df['intent'].value_counts())
    print("\nTest Intent Dağılımı:")
    print(test_df['intent'].value_counts())
    
    # Kaydet
    train_df.to_excel('data/train_dataset.xlsx', index=False)
    test_df.to_excel('data/test_dataset.xlsx', index=False)
    
    print("\n✓ Dosyalar kaydedildi:")
    print("  - data/train_dataset.xlsx")
    print("  - data/test_dataset.xlsx")
    
    return train_df, test_df

if __name__ == "__main__":
    print("="*60)
    print("VERİ HAZIRLAMA BAŞLIYOR")
    print("="*60)
    
    # Veri setini hazırla
    df = prepare_dataset('data/chatbot_dataset.xlsx')
    
    # Train-test split
    train_df, test_df = split_data(df)
    
    print("\n" + "="*60)
    print("VERİ HAZIRLAMA TAMAMLANDI!")
    print("="*60)