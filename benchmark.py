# benchmark.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.groq_model import GroqChatbotRAG
from models.mistral_model import MistralChatbot
from dotenv import load_dotenv
import os

load_dotenv()

def run_benchmark():
    """Her iki modeli test et ve karşılaştır"""
    
    # Veriyi yükle
    test_df = pd.read_excel('data/test_dataset.xlsx')
    train_df = pd.read_excel('data/train_dataset.xlsx')
    
    print("="*60)
    print("BENCHMARK BAŞLIYOR")
    print("="*60)
    
    # Groq testi
    print("\n1. GROQ MODELİ TEST EDİLİYOR...")
    print("-"*60)
    groq_bot = GroqChatbotRAG()
    groq_results = groq_bot.evaluate_model(test_df, train_df, num_few_shot=10)
    
    # Mistral testi
    print("\n2. MISTRAL MODELİ TEST EDİLİYOR...")
    print("-"*60)
    mistral_bot = MistralChatbot()
    mistral_results = mistral_bot.evaluate_model(test_df, train_df, num_few_shot=10)
    
    # Karşılaştırma tablosu
    comparison_df = pd.DataFrame({
        'Model': ['Groq (Mixtral)', 'Mistral AI'],
        'Precision': [groq_results['precision'], mistral_results['precision']],
        'Recall': [groq_results['recall'], mistral_results['recall']],
        'F1 Score': [groq_results['f1_score'], mistral_results['f1_score']]
    })
    
    print("\n" + "="*60)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Kaydet
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    # Görselleştirme
    visualize_results(groq_results, mistral_results, comparison_df)
    
    return groq_results, mistral_results, comparison_df

def visualize_results(groq_results, mistral_results, comparison_df):
    """Sonuçları görselleştir"""
    
    # 1. Metrik karşılaştırması (Bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Precision', 'Recall', 'F1 Score']
    for idx, metric in enumerate(metrics):
        axes[idx].bar(comparison_df['Model'], comparison_df[metric], color=['#FF6B6B', '#4ECDC4'])
        axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Değerleri bar üzerine yaz
        for i, v in enumerate(comparison_df[metric]):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Metrik karşılaştırma grafiği kaydedildi: results/metrics_comparison.png")
    
    # 2. Confusion Matrix - Groq
    plt.figure(figsize=(10, 8))
    intents = ["greeting", "order_dessert", "ask_recommendation", "check_ingredients", "goodbye"]
    cm_groq = np.array(groq_results['confusion_matrix'])
    
    sns.heatmap(cm_groq, annot=True, fmt='d', cmap='Blues', 
                xticklabels=intents, yticklabels=intents,
                cbar_kws={'label': 'Count'})
    plt.title('Groq (Mixtral) - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/groq_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Groq confusion matrix kaydedildi: results/groq_confusion_matrix.png")
    
    # 3. Confusion Matrix - Mistral
    plt.figure(figsize=(10, 8))
    cm_mistral = np.array(mistral_results['confusion_matrix'])
    
    sns.heatmap(cm_mistral, annot=True, fmt='d', cmap='Greens',
                xticklabels=intents, yticklabels=intents,
                cbar_kws={'label': 'Count'})
    plt.title('Mistral AI - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/mistral_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Mistral confusion matrix kaydedildi: results/mistral_confusion_matrix.png")
    
    plt.close('all')

if __name__ == "__main__":
    # Results klasörünü oluştur
    os.makedirs('results', exist_ok=True)
    
    # Benchmark çalıştır
    groq_results, mistral_results, comparison = run_benchmark()
    
    print("\n" + "="*60)
    print("BENCHMARK TAMAMLANDI!")
    print("="*60)
    print("\nSonuçlar 'results/' klasörüne kaydedildi.")