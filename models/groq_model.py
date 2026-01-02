import os
import json
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Optional, Tuple
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# SINIF ADI DÜZELTİLDİ: GroqChatbotRAG
class GroqChatbotRAG:
    def __init__(self, api_key: Optional[str] = None, train_df: Optional[pd.DataFrame] = None):
        """
        Groq API ve RAG altyapısını başlatan sınıf.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            # Hata fırlatmak yerine None dönelim ki Streamlit çökmesin, uyarı versin
            print("UYARI: GROQ_API_KEY bulunamadı!")
            
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
        self.intents = [
            "greeting", "order_dessert", "ask_recommendation",
            "check_ingredients", "goodbye"
        ]
        
        # Embedding modelini bir kere yükle
        print("Model yükleniyor: paraphrase-multilingual-MiniLM-L12-v2...")
        try:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            print(f"Embedding modeli yüklenemedi: {e}")
            self.embedding_model = None
        
        # Vektör veritabanı değişkenleri
        self.train_data = None
        self.index = None
        
        if train_df is not None and self.embedding_model is not None:
            self.setup_vector_db(train_df)

    def setup_vector_db(self, train_df: pd.DataFrame):
        """Eğitim verilerini vektör veritabanına işler."""
        print("Vektör veritabanı oluşturuluyor...")
        self.train_data = train_df.copy()
        
        # Metinleri vektöre çevir
        texts = self.train_data['text'].tolist()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # FAISS index oluştur
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ {len(texts)} örnek başarıyla indekslendi.")

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Query'e en benzer eğitim verilerini getirir (Few-Shot Learning için).
        """
        if self.index is None:
            return ""
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        similar_rows = self.train_data.iloc[indices[0]]
        
        context_str = "\nReferans Örnekler:\n"
        for _, row in similar_rows.iterrows():
            context_str += f"- Kullanıcı: '{row['text']}' -> Niyet: {row['intent']}\n"
            
        return context_str

    def predict_intent(self, user_message: str) -> str:
        """
        RAG destekli niyet tahmini yapar.
        """
        # Benzer örnekleri çek (RAG Step)
        context_examples = self.retrieve_context(user_message, k=5)
        
        system_prompt = f"""Sen bir sınıflandırma asistanısın. Aşağıdaki mesajın niyetini (intent) belirle.

Kategoriler: {', '.join(self.intents)}

{context_examples}

Sadece kategori ismini yaz, başka hiçbir şey yazma."""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Mesaj: {user_message}\nNiyet:"}
                ],
                model=self.model,
                temperature=0.0, 
                max_tokens=10
            )
            
            predicted = response.choices[0].message.content.strip().lower()
            
            for intent in self.intents:
                if intent in predicted:
                    return intent
            return "unknown"
            
        except Exception as e:
            print(f"Intent tahmini hatası: {e}")
            return "error"

    def chat(self, user_message: str, conversation_history: List[Dict] = None) -> Tuple[str, str]:
        """
        Sohbet fonksiyonu. Hafıza (history) kullanır.
        """
        if conversation_history is None:
            conversation_history = []

        # 1. Niyeti belirle
        intent = self.predict_intent(user_message)
        
        # 2. Sistem Promptunu Hazırla
        system_prompt = f"""Sen 'Tatlı Rüyalar' adında bir tatlı mağazasının yapay zeka asistanısın.
Tespit edilen kullanıcı niyeti: {intent.upper()}

Kurallar:
1. Çok nazik, samimi ve iştah açıcı konuş.
2. Sadece tatlılar, içecekler ve mağaza hakkında konuş.
3. Eğer niyet 'order_dessert' ise siparişi onayla ve başka bir isteği olup olmadığını sor.
4. Yanıtların kısa ve öz olsun (maksimum 3 cümle).

Menüden Örnekler: Fıstıklı Baklava, Sütlaç, San Sebastian Cheesecake, Tiramisu."""

        # 3. Mesaj geçmişini hazırla
        messages = [{"role": "system", "content": system_prompt}]
        
        # Eski konuşmaları ekle (Son 4 mesajı tutmak token tasarrufu sağlar)
        # History formatı: {'role': 'user', 'content': '...'}
        messages.extend(conversation_history[-4:]) 
        
        # Yeni mesajı ekle
        messages.append({"role": "user", "content": user_message})

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=150
            )
            
            response = chat_completion.choices[0].message.content.strip()
            return response, intent
            
        except Exception as e:
            return f"Hata oluştu: {e}", intent