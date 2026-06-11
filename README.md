# AgriBot 🌱🤖

**A Hybrid Multimodal Decision Support System for Precision Agriculture**

AgriBot is an advanced, AI-enabled agricultural advisory platform designed to democratize precision farming. It bridges the gap between complex machine learning algorithms and accessibility by integrating a conversational Generative AI interface with strict deterministic routing, computer vision, and multilingual neural text-to-speech (TTS).

## ✨ Key Features

* **Deterministic Pre-Routing:** Prevents AI "hallucinations" by intercepting strict numerical soil/weather data (N, P, K, pH, etc.) and forcing it through mathematically precise predictive models.
* **Precision ML Predictions:** Utilizes XGBoost and Random Forest algorithms for highly accurate Crop Recommendations and Yield Predictions.
* **Visual Disease Diagnostics:** Features an integrated ResNet50 Convolutional Neural Network (CNN) to instantly classify plant diseases from user-uploaded leaf images.
* **State-Aware Conversational Memory:** Uses Django session management to remember active farm parameters across multi-turn chats (e.g., asking for fertilizer recommendations right after a crop prediction).
* **Retrieval-Augmented Generation (RAG):** General agronomic queries are factually grounded using a FAISS vector database loaded with vetted agricultural literature.
* **Multilingual Audio Accessibility:** Breaks the literacy barrier by translating and synthesizing technical advice into accelerated (1.5x) regional language audio streams (English, Hindi, Kannada) using Microsoft Edge Neural TTS.

## 🏗️ System Architecture

```text
[ INPUT ] User Input (Text / Image) 
    ⬇ 
[ SERVER ] Django Backend 
    ⬇ 
[ ROUTING ] Deterministic Router 
    ⬋                ⬇                ⬊ 
(If Image)      (If 7 Params)      (If Generic Chat) 
[ ResNet50 ]    [ XGBoost / RF ]    [ FAISS Vector DB ] 
    ⬊                ⬇                ⬋ 
[ SYNTHESIS ] Gemini 2.5 Flash LLM 
    ⬇ 
[ LOCALIZATION ] Language Translation 
    ⬇ 
[ AUDIO GENERATION ] Microsoft Edge TTS 
    ⬇ 
[ OUTPUT ] Farmer Audio & Text Output

```

## 💻 Tech Stack

* **Backend Framework:** Python 3.10+, Django 5.x
* **Machine Learning (Tabular):** Scikit-Learn, XGBoost, Pandas, NumPy
* **Deep Learning (Vision):** TensorFlow 2.x, Keras (ResNet50)
* **Generative AI & NLP:** Google Gemini 2.5 Flash API, LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search), HuggingFace Embeddings
* **Audio Generation:** `edge-tts` (Microsoft Azure Neural Voices), `asyncio`

## 🚀 Installation and Setup

### Prerequisites

* Python 3.10 or higher
* Git

### Step-by-Step Guide

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/agribot.git
cd agribot

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install the required dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up Environment Variables:**
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here

```


5. **Ensure ML Models are in place:**
Verify that your `ml_models/` directory contains all necessary serialized models (`.pkl` and `.h5` files) and the FAISS index folder.
6. **Run the Django Development Server:**
```bash
python manage.py runserver

```


Navigate to `http://127.0.0.1:8000/` in your web browser.

## 📖 Usage

* **Text Chat:** Simply type your agricultural queries. To trigger the ML models directly, provide the 7 critical parameters in your text (e.g., "N=90, P=42, K=43, temp=20.8, humidity=82, ph=6.5, rainfall=202").
* **Disease Detection:** Use the upload button to submit a photo of a diseased plant leaf. The ResNet50 model will analyze it and Gemini will provide treatment steps.
* **Voice Playback:** Ensure your browser allows audio playback. Select your preferred language (e.g., Kannada, Hindi) from the UI dropdown to hear the Neural TTS response.

## 👥 Team & Roles

* **VIVEK V RON:** Backend Development
* **VIVEK TUBAKI:** Backend Development
* **VISHWANATH REDDY:** Frontend Development & Testing
* **VISHWAS HAVALADA:** Frontend Development & Testing

## 👨‍💻 Author

**Vivek V Ron** *Undergraduate Student, Computer Science and Engineering*
*MS Engineering College, Bengaluru*
