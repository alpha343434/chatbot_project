import streamlit as st
import sys
import os
import pandas as pd 
from dotenv import load_dotenv

# --- YOL AYARI (PATH CONFIGURATION) ---
# Mevcut dosyanın (streamlit_app.py) bulunduğu klasörü bul
current_dir = os.path.dirname(os.path.abspath(__file__))
# Bir üst klasöre (CHATBOT_PROJECT) çık
parent_dir = os.path.dirname(current_dir)
# Bu üst klasörü Python'un arama yollarına ekle
sys.path.append(parent_dir)

# Artık models klasörü sorunsuz import edilebilir
from models.groq_model import GroqChatbotRAG
from models.mistral_model import MistralChatbot
from dotenv import load_dotenv

# .env yükle
load_dotenv()

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Tatlış Chatbot",
    page_icon="🧁",
    layout="centered"
)

# --- CSS (Sadece Intent Badge için minimal stil) ---
st.markdown("""
<style>
    .stDeployButton {display:none;}
    .intent-badge {
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #f0f2f6;
        color: #31333f;
        border: 1px solid #d0d2d6;
        margin-left: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME (CACHE) ---
# Bu fonksiyonlar sadece bir kez çalışır, her tıklamada modeli tekrar yüklemez.
@st.cache_resource
def load_groq_model():
    try:
        # Veri setini yükle (RAG için gerekli)
        if os.path.exists('data/train_dataset.xlsx'):
            df = pd.read_excel('data/train_dataset.xlsx')
            return GroqChatbotRAG(train_df=df)
        else:
            st.error("⚠️ data/train_dataset.xlsx bulunamadı! Groq RAG çalışmayabilir.")
            return GroqChatbotRAG() # Boş başlat
    except Exception as e:
        st.error(f"Groq yüklenirken hata: {e}")
        return None

@st.cache_resource
def load_mistral_model():
    try:
        # Mistral için de train verisini yükleyelim ki Few-Shot yapabilsin
        if os.path.exists('data/train_dataset.xlsx'):
            df = pd.read_excel('data/train_dataset.xlsx')
            return MistralChatbot(train_df=df)
        return MistralChatbot()
    except Exception as e:
        st.error(f"Mistral yüklenirken hata: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    selected_model_name = st.radio(
        "Model Seçimi:",
        ["Groq (Llama 3.3)", "Mistral (Open Mistral 7B)"],
        captions=["Hızlı & RAG Destekli", "Hafif & Hızlı"]
    )
    
    st.markdown("---")
    st.markdown("### Intent Rehberi")
    st.caption("Botun anladığı niyetler:")
    st.code("""
greeting: Merhaba/Selam
order_dessert: Sipariş
ask_rec: Öneri İsteme
check_ing: İçerik Sorma
goodbye: Vedalaşma
    """, language="yaml")
    
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- INIT STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Modelleri yükle
groq_bot = load_groq_model()
mistral_bot = load_mistral_model()

# --- ANA ARAYÜZ ---
st.title("🧁 Tatlış Chatbot")
st.caption("Size en tatlı anlarınızda eşlik eden yapay zeka asistanı.")

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Eğer asistansa ve intent bilgisi varsa göster
        if message["role"] == "assistant" and "intent" in message:
             st.markdown(f'<span class="intent-badge">Intent: {message["intent"]}</span>', unsafe_allow_html=True)

# --- CHAT INPUT & MANTIK ---
if prompt := st.chat_input("Hangi tatlıyı istersiniz?"):
    
    # 1. Kullanıcı mesajını ekrana bas ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Bot yanıtı için alan aç
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        # Seçili botu belirle
        active_bot = None
        current_model_tag = ""
        
        if "Groq" in selected_model_name:
            active_bot = groq_bot
            current_model_tag = "Groq"
        else:
            active_bot = mistral_bot
            current_model_tag = "Mistral"
            
        if active_bot:
            with st.spinner(f"{current_model_tag} düşünüyor..."):
                # Sohbet geçmişini modele uygun formata getir (Groq için)
                history_for_model = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages 
                    if m["role"] != "system"
                ]
                
                # Yanıt al
                # Not: Her iki modelinizin chat fonksiyonu (response, intent) döndürmeli
                response_text, intent = active_bot.chat(prompt, conversation_history=history_for_model)
                
                # Ekrana bas
                message_placeholder.markdown(response_text)
                st.markdown(f'<span class="intent-badge">Intent: {intent}</span>', unsafe_allow_html=True)
                
                # Geçmişe kaydet
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "intent": intent,
                    "model": current_model_tag
                })
        else:
            st.error("Seçilen model başlatılamadı.")