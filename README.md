# 🍰 Tatlış Chatbot: AI Destekli Tatlı Mağazası Asistanı

Bu proje, bir tatlı mağazası için geliştirilmiş, **LLM (Büyük Dil Modelleri)** tabanlı akıllı bir sohbet asistanıdır. Proje kapsamında **Groq** ve **Mistral AI** modelleri karşılaştırmalı olarak kullanılmış; **RAG (Retrieval-Augmented Generation)** ve **Few-Shot Learning** teknikleri uygulanmıştır.

## 🚀 Projenin Amacı

Bu ödev projesinin temel amaçları şunlardır:
1.  Kullanıcı niyetlerini (Intent Classification) doğru tespit etmek.
2.  Mağaza menüsü ve tatlı içerikleri hakkında doğru bilgiler vermek.
3.  Farklı LLM mimarilerinin (Llama 3.3 vs Mistral Nemo) performansını kıyaslamak.
4.  RAG (Vektör tabanlı) ve Few-Shot (Örnek tabanlı) yaklaşımlarını pratikte uygulamak.

## 🛠️ Kullanılan Teknolojiler ve Yöntemler

* **Arayüz:** Streamlit
* **Dil:** Python 3.10+
* **Model 1 (RAG):** Groq API (Llama-3.3-70b-versatile) + FAISS + SentenceTransformers
* **Model 2 (Few-Shot):** Mistral API (Open-Mistral-Nemo)
* **Veri İşleme:** Pandas, Scikit-learn (Performans metrikleri için)

### Niyet Sınıflandırma Kategorileri
Bot aşağıdaki 5 niyeti anlayacak şekilde eğitilmiştir:
* `greeting`: Selamlama
* `order_dessert`: Tatlı siparişi
* `ask_recommendation`: Öneri isteme
* `check_ingredients`: İçerik/Malzeme sorma
* `goodbye`: Vedalaşma

## 📂 Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADINIZ/tatli-magazasi-chatbot.git](https://github.com/KULLANICI_ADINIZ/tatli-magazasi-chatbot.git)
cd tatli-magazasi-chatbot