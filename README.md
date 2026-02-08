# 🎵 Music Intelligence System (Dataset Edition)

> **Advanced AI Lyrics Analysis trained on Real Dataset**
> *Sentiment (VADER) • Artist Recognition (Gradient Boosting) • Language ID*

## � Dataset Instructions
This system is designed to work with the **Chords and Lyrics Dataset** from Kaggle.

1.  **Download Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/eitanbentora/chords-and-lyrics-dataset)
2.  **Upload to Colab**: Rename the file to `lyrics_scraped.csv` (or keep original name) and upload it to the root folder of your Colab runtime.
3.  **Run**: The script will automatically detect the CSV, clean the chords (e.g., `[Am]`), and train the models.

## 🚀 Features
- **Specific Artist Recognition**: Trains an advanced `GradientBoostingClassifier` to recognize the writing style of the **Top 10 Artists** in your dataset.
- **Sentiment Analysis**: Uses **NLTK VADER**, the gold standard for text sentiment, to categorize lyrics as Positive, Negative, or Neutral.
- **Language Detection**: Automatically Identifies the language of the song.
- **Smart Cleaning**: Includes a regex pre-processor that strips out musical chords (`[C#m]`, etc.) to focus purely on the poetry.

## 🛠️ Architecture
- **Model**: Gradient Boosting (sklearn)
- **Features**: TF-IDF (1-2 ngrams, 5000 features)
- **UI**: Gradio (Web Interface)

## 🏃 How to Run
Run `python music_intelligence_advanced.py` or paste the code into Google Colab.
