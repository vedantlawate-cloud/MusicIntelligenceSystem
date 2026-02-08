# Music Intelligence System - Dataset Edition (Language & Emotion)
# -----------------------------------------------------------------------------
# Instructions:
# 1. Upload 'kaggle.json' to Colab Files.
# 2. Run this script.
# 3. It will load `eitanbentora/chords-and-lyrics-dataset` (optional now, but good for corpus).
# -----------------------------------------------------------------------------

# --- 1. Dependencies & Setup ---
import subprocess
import sys
import os
import json
import shutil

def install_packages():
    packages = ["pandas", "scikit-learn", "nltk", "langdetect", "gradio", "matplotlib", "seaborn"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

import pandas as pd
import numpy as np
import re
import glob
import zipfile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
import gradio as gr

# Setup NLTK
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()
DetectorFactory.seed = 0

# --- 2. Kaggle Setup (Optional but kept for Dataset Edition feel) ---
def setup_dataset():
    print("🚀 Setting up Environment...")
    
    # 1. Config Check
    config_path = "kaggle.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                creds = json.load(f)
                os.environ['KAGGLE_USERNAME'] = creds['username']
                os.environ['KAGGLE_KEY'] = creds['key']
            print("✅ Kaggle Credentials Loaded.")
            
            # 2. Install/Import Kaggle
            try: import kaggle
            except: 
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                import kaggle
            
            # 3. Download
            print("📥 Downloading Lyrics Corpus...")
            subprocess.check_call(["kaggle", "datasets", "download", "-d", "eitanbentora/chords-and-lyrics-dataset", "--force"])
            
            with zipfile.ZipFile("chords-and-lyrics-dataset.zip", 'r') as z:
                z.extractall(".")
            print("✅ Dataset Extracted.")
            
        except Exception as e:
            print(f"⚠️ Dataset setup skipped: {e}")
            print("💡 System will run in Inference-Only Mode (Perfectly fine for Emotion/Language).")

setup_dataset()

# --- 3. Analysis Logic ---

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"\[.*?\]", "", text) # Remove chords
    text = re.sub(r"\(.*?\)", "", text) # Remove parens
    return text.strip()

def analyze_music(lyrics):
    if not lyrics or not lyrics.strip():
        return "⚠️ Please enter text.", None
    
    # A. Cleaning
    clean_lyrics = clean_text(lyrics)
    
    # B. Language Detection
    try:
        lang_code = detect(clean_lyrics)
        lang_map = {
            "en": "🇬🇧 English", "es": "🇪🇸 Spanish", "fr": "🇫🇷 French", 
            "de": "🇩🇪 German", "it": "🇮🇹 Italian", "pt": "🇵🇹 Portuguese",
            "ko": "🇰🇷 Korean", "ja": "🇯🇵 Japanese"
        }
        language_display = lang_map.get(lang_code, f"🌐 {lang_code.upper()}")
    except:
        language_display = "❓ Unknown"

    # C. Emotion / Sentiment (VADER)
    scores = sia.polarity_scores(clean_lyrics)
    compound = scores['compound']
    
    # Map Compound Score to Emotion Label
    if compound >= 0.8:
        emotion = "🤩 Euphoric / Very Happy"
        color = "green"
    elif 0.4 <= compound < 0.8:
        emotion = "🙂 Happy / Positive"
        color = "lightgreen"
    elif -0.2 < compound < 0.4:
        emotion = "😐 Neutral / Calm"
        color = "gray"
    elif -0.6 < compound <= -0.2:
        emotion = "😟 Sad / Melancholic"
        color = "orange"
    else:
        emotion = "😭 Depressed / Very Negative"
        color = "red"

    # Create Report
    report = (
        f"### 🌍 **Language**\n"
        f"# {language_display}\n\n"
        f"### 🎭 **Emotion Detected**\n"
        f"## {emotion}\n"
        f"*Intensity Score: {compound:.2f}* (Scale: -1.0 to +1.0)\n"
    )
    
    # Chart Data for UI (Pos/Neu/Neg breakdown)
    chart_data = {"Positive": scores['pos'], "Neutral": scores['neu'], "Negative": scores['neg']}
    
    return report, chart_data

# --- 4. Refined UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎵 Music Intelligence System
        **Language & Emotion Analysis Engine**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_lyrics = gr.Textbox(
                lines=8, 
                placeholder="Paste song lyrics here...", 
                label="Input Lyrics"
            )
            btn = gr.Button("🔍 Analyze Song", variant="primary")
            gr.Markdown("*Tip: You can paste lyrics in any language.*")
            
        with gr.Column(scale=1):
            out_report = gr.Markdown(label="Analysis Results")
            out_chart = gr.Label(label="Sentiment Composition")

    btn.click(
        fn=analyze_music, 
        inputs=input_lyrics, 
        outputs=[out_report, out_chart]
    )

print("✅ System Ready. Launching UI...")
demo.launch(share=True)
