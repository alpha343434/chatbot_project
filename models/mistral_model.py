import os
import json
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
from mistralai import Mistral
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

class MistralChatbot:
    def __init__(self, api_key: Optional[str] = None, train_df: Optional[pd.DataFrame] = None):
        """
        Mistral Chatbot Başlatıcı
        
        Args:
            api_key: Mistral API anahtarı
            train_df: Few-shot learning için kullanılacak eğitim verisi
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        
        if not self.api_key:
            print("UYARI: MISTRAL_API_KEY bulunamadı! .env dosyasını kontrol edin.")
            self.client = None
        else:
            self.client = Mistral(api_key=self.api_key)
            
        self.model = "open-mistral-nemo" 
        
        self.intents = [
            "greeting", "order_dessert", "ask_recommendation", 
            "check_ingredients", "goodbye"
        ]
        
        # --- STATIC FEW-SHOT HAZIRLIĞI ---
        # RAG kullanmadığımız için, her intent'ten 2-3 örnek seçip 
        # bunları sabit prompt olarak modele vereceğiz.
        self.few_shot_context = ""
        if train_df is not None:
            self._prepare_static_examples(train_df)
            
        print(f"✓ Mistral API başlatıldı (Model: {self.model})")

    def _prepare_static_examples(self, df: pd.DataFrame, samples_per_intent: int = 2):
        """Verisetinden her kategori için sabit örnekler seçer."""
        try:
            examples_str = "\nREFERANS ÖRNEKLER:\n"
            
            # Sütun isimlerini normalize et
            df.columns = df.columns.str.lower().str.strip()
            
            for intent in self.intents:
                # O intent'e ait örnekleri filtrele
                intent_rows = df[df['intent'] == intent]
                
                # Eğer yeterli örnek varsa rastgele seç, yoksa hepsini al
                if len(intent_rows) >= samples_per_intent:
                    sampled = intent_rows.sample(samples_per_intent)
                else:
                    sampled = intent_rows
                
                for _, row in sampled.iterrows():
                    examples_str += f"- Kullanıcı: '{row['text']}' -> Intent: {intent}\n"
            
            self.few_shot_context = examples_str
            print(f"✓ {len(self.intents) * samples_per_intent} adet sabit örnek yüklendi.")
            
        except Exception as e:
            print(f"Örnek hazırlama hatası: {e}")
            self.few_shot_context = ""

    def predict_intent(self, user_message: str) -> str:
        """Kullanıcı mesajının niyetini tahmin eder."""
        if not self.client: return "error"

        system_prompt = f"""Sen bir sınıflandırma motorusun. 
Görevin: Kullanıcı mesajını aşağıdaki kategorilerden birine eşleştirmek.

KATEGORİLER:
{', '.join(self.intents)}

{self.few_shot_context}

KURALLAR:
1. Sadece kategori ismini (intent) yaz.
2. Açıklama yapma, noktalama işareti koyma.
3. Mesaj şunlardan birine tam uymuyorsa en yakını seç.

Mesaj: "{user_message}"
Intent:"""

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.0, # Tutarlılık için 0
                max_tokens=10
            )
            
            predicted = response.choices[0].message.content.strip().lower()
            
            # Temizlik
            predicted = predicted.replace("intent:", "").strip()
            
            # Doğrulama
            for intent in self.intents:
                if intent in predicted:
                    return intent
            
            return "unknown"

        except Exception as e:
            print(f"Intent Error: {e}")
            return "error"

    def chat(self, user_message: str, conversation_history: List[Dict] = None) -> Tuple[str, str]:
        """Ana sohbet fonksiyonu (Hafıza destekli)."""
        if not self.client: return "API Key Eksik", "error"
        
        # 1. Niyeti Belirle
        intent = self.predict_intent(user_message)
        
        # 2. Yanıt Üretme Prompt'u
        system_instructions = f"""Sen 'Tatlı Rüyalar' pastanesinin yapay zeka asistanısın.
Tespit edilen kullanıcı niyeti: {intent.upper()}

Kurallar:
1. Çok nazik, samimi ve iştah açıcı konuş.
2. Sadece tatlılar, içecekler ve mağaza hakkında konuş.
3. Eğer niyet 'order_dessert' ise siparişi onayla ve başka bir isteği olup olmadığını sor.
4. Yanıtların kısa ve öz olsun (maksimum 3 cümle).

Menüden Örnekler: Fıstıklı Baklava, Sütlaç, San Sebastian Cheesecake, Tiramisu.
"""
        
        # Mesaj geçmişini hazırla
        messages = [{"role": "system", "content": system_instructions}]
        
        # Geçmişi ekle (Son 4 mesaj - token tasarrufu için)
        if conversation_history:
            messages.extend(conversation_history[-4:])
            
        # Yeni mesajı ekle
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.7, # Yaratıcılık için
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip(), intent
            
        except Exception as e:
            return f"Şu an fırın çok sıcak, yanıt veremiyorum: {e}", intent

    def evaluate_model(self, test_df: pd.DataFrame):
        """Model başarısını test seti üzerinde ölçer."""
        print(f"\nDeğerlendirme Başlıyor: {len(test_df)} örnek...")
        
        y_true = []
        y_pred = []
        
        for i, row in test_df.iterrows():
            if i % 10 == 0: print(f"İşleniyor: {i}/{len(test_df)}")
            
            pred = self.predict_intent(row['text'])
            y_true.append(row['intent'])
            y_pred.append(pred)
            
            # API limitine takılmamak için minik bekleme
            time.sleep(0.2)
            
        print("\n" + "="*50)
        print("MISTRAL SONUÇ RAPORU")
        print("="*50)
        
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        
        # Metrikleri döndür
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'model': 'Mistral (Static Few-Shot)',
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'predictions': y_pred,
            'true_labels': y_true
        }

# --- TEST BLOĞU ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Veri setini yükle (Varsa)
    if os.path.exists('data/train_dataset.xlsx'):
        print("Veriseti yükleniyor...")
        df = pd.read_excel('data/train_dataset.xlsx')
        bot = MistralChatbot(train_df=df)
    else:
        print("Veriseti bulunamadı, boş başlatılıyor.")
        bot = MistralChatbot()
        
    # Manuel Sohbet Testi
    history = []
    print("\n--- SOHBET (Çıkış: q) ---")
    while True:
        msg = input("Siz: ")
        if msg.lower() == 'q': break
        
        resp, intent = bot.chat(msg, history)
        
        print(f"Bot ({intent}): {resp}")
        
        # History güncelle
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": resp})