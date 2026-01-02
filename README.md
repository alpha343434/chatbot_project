# 🍰 Tatlış Chatbot: AI Destekli Tatlı Mağazası Asistanı

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Models](https://img.shields.io/badge/LLMs-Groq%20%26%20Mistral-green)

**Tatlış Chatbot**, tatlı severler için geliştirilmiş, sipariş alabilen, ürün içerikleri hakkında bilgi veren ve kişiselleştirilmiş öneriler sunan yapay zeka tabanlı bir asistandır.

Bu proje, **Doğal Dil İşleme (NLP)** alanında iki farklı yaklaşımı (**RAG** ve **Few-Shot Learning**) karşılaştırmak ve performanslarını analiz etmek amacıyla geliştirilmiştir.

## 🚀 Projenin Amacı

* Müşteri niyetlerini (Intent Classification) %90+ doğrulukla tespit etmek.
* Farklı LLM mimarilerinin (Llama 3.3 vs Mistral Nemo) performansını kıyaslamak.
* Vektör tabanlı arama (RAG) ile örnek tabanlı öğrenme (Few-Shot) arasındaki farkları analiz etmek.

## 🧠 Kullanılan Modeller ve Yöntemler

Projede iki farklı "Agent" mimarisi tasarlanmıştır:

### 1. Model A: Groq (Llama 3.3) + RAG
* **Teknoloji:** Groq API, FAISS, SentenceTransformers.
* **Yöntem (RAG):** Kullanıcı sorusu vektöre çevrilir ve veri tabanındaki en benzer geçmiş diyaloglar bulunarak modele "bağlam" (context) olarak verilir.
* **Avantajı:** Geniş veri setlerinde (örneğin 1000+ ürünlü menü) çok daha tutarlı cevaplar verir.

### 2. Model B: Mistral (Nemo) + Few-Shot Learning
* **Teknoloji:** Mistral AI API.
* **Yöntem (Few-Shot):** Eğitim setinden rastgele seçilen 2-3 örnek diyalog, modelin sistem mesajına (System Prompt) dinamik olarak eklenir.
* **Avantajı:** Hızlı kurulum, düşük gecikme süresi (latency) ve yüksek genelleme yeteneği.

## 📂 Proje Yapısı

```bash
tatli-magazasi-chatbot/
├── app/
│   └── streamlit_app.py      # Kullanıcı Arayüzü
├── data/
│   ├── train_dataset.xlsx    # Eğitim Verisi (800+ satır)
│   └── test_dataset.xlsx     # Test Verisi (200+ satır)
├── models/
│   ├── groq_model.py         # RAG Modeli
│   └── mistral_model.py      # Few-Shot Modeli
├── results/                  # Analiz Grafikleri
│   ├── metrics_comparison.png
│   └── comparison.csv
├── benchmark.py              # Performans Test Kodu
├── requirements.txt          # Kütüphaneler
└── README.md                 # Dökümantasyon

### 📊 Model Performans Karşılaştırması

Aşağıdaki tablo, test veri seti üzerinde yapılan benchmark sonuçlarını göstermektedir:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Groq (Mixtral 8x7B) + RAG** | 0.8475 | 0.7820 | 0.7893 |
| **Mistral AI (Nemo) + Few-Shot** | 0.7622 | 0.2085 | 0.3224 |

> **Analiz:** Groq modeli, RAG mimarisi sayesinde niyetleri (intents) yakalamada (Recall) ve genel doğrulukta (F1 Score) Mistral modeline göre belirgin bir üstünlük sağlamıştır. Mistral modeli, sınırlı örnek (Few-Shot) ile çalıştığı için bazı niyetleri kaçırmış (düşük Recall) ancak tahmin ettiğinde nispeten yüksek doğruluk (Precision) sergilemiştir.