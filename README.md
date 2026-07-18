# AgriBot | Intelligent AI Assistant for Farming 🌾

AgriBot is a state-of-the-art agricultural web platform designed to empower farmers and agricultural enthusiasts. Built with a sleek, minimalist **Dark AI SaaS UI**, AgriBot leverages advanced Machine Learning, Deep Learning, and Dual-LLM architecture to deliver hyper-accurate crop recommendations, fertilizer analysis, yield predictions, and plant disease diagnostics.

## 🚀 Key Features

*   **Multimodal Security Gatekeeper:** Employs Google Gemini 2.5 Flash to pre-validate image uploads, ensuring only actual plant/leaf images are passed to the deep learning models.
*   **Dual-LLM Fallback Architecture:** Guarantees 99.9% uptime by automatically failing over from primary (Gemini 2.5 Flash) to secondary (Groq Llama 3) if rate limits or downtimes occur.
*   **Predictive ML Pipelines:** Uses XGBoost for crop classification, Random Forest for yield regressions, and custom tree models for precise fertilizer blends based on N-P-K limits.
*   **Plant Pathology Deep Learning:** A ResNet50 Convolutional Neural Network (CNN) detects diseases from leaf scans with a 65% confidence threshold guardrail.
*   **RAG Knowledge Base:** FAISS vector database coupled with MiniLM-L6-v2 embeddings for answering complex agricultural queries from technical documents.
*   **Premium Edge TTS:** Instant, high-fidelity text-to-speech feedback using native Windows async-wrapped Microsoft Edge Neural voices (English, Hindi, Kannada).
*   **Native Multi-Language Support:** Seamlessly switch the entire UI to Hindi or Kannada using an integrated custom translation widget.

## 🧠 High Level Architecture

```text
+---------------------------------------------------------------------------------------------------+
|                                           USER INTERFACE                                          |
|  +---------------------------+  +---------------------------+  +-------------------------------+  |
|  |     Forms Interface       |  |     Plant Disease Cam     |  |            AgriBot            |  |
|  | (Crop/Fertilizer/Yield)   |  |     Image Upload Widget   |  |          Chat Window          |  |
|  +-------------+-------------+  +-------------+-------------+  +---------------+---------------+  |
+----------------|------------------------------|--------------------------------|------------------+
                 | (POST Form Data)             | (POST Multipart Image)         | (POST Ajax JSON)
                 v                              v                                v
+---------------------------------------------------------------------------------------------------+
|                                         DJANGO BACKEND                                            |
|                                                                                                   |
|   [ views.py ] <==============================================================================+   |
|        |                                                                                      |   |
|        |--> 1. ROUTING & PARSING INTERCEPT                                                    |   |
|        |    • Session Param Accumulator & Memory Checker                                      |   |
|        |    • Deterministic RegEx Feature Extractor                                           |   |
|        |    • Null-Safe Guardrail Validator (Limits Check)                                    |   |
|        |                                                                                      |   |
|        +--> 2. PRE-DICTION VERIFICATION GATES                                                 |   |
|        |    • Multimodal Leaf / Non-Leaf Image Validator                                      |   |
|        |                                                                                      |   |
|        +--> 3. DUAL-LLM TEXT GENERATION ROUTER                                                |   |
|             • Primary: Gemini 2.5 Flash API (Contextual RAG Prompts)                          |   |
|             • Secondary Fallback: Groq Cloud API (Llama 3.1 8B Instant)                       |   |
+---------------------------------------------------------------------------------------|-----------+
         |                                      |                                       |
         | (Numerical Array / Categorical)      | (Preprocessed Array)                  | (Vector Query)
         v                                      v                                       v
+-----------------------------+       +--------------------+                  +--------------------+
|    PREDICTIVE PIPELINES     |       |  PLANT PATHOLOGY   |                  |    FAISS VECTOR    |
|       (ML & DATA)           |       |   PIPELINE (DL)    |                  |   KNOWLEDGE BASE   |
|                             |       |                    |                  |                    |
|  • XGBoost Crop Classifier  |       |  • ResNet50 CNN    |                  |  • MiniLM-L6-v2    |
|  • Fertilizer Tree Model    |       |  • Confidence Gate |                  |    Embeddings      |
|  • Random Forest Regressor  |       |    Threshold (65%) |                  |  • NLP Intent      |
|    (Yield Model - hg/ha)    |       +--------------------+                  |    Classifier      |
|  • One-Hot Column Encoders  |                                               +--------------------+
|    (Area & Item Mapping)    |                                                         |
+-----------------------------+                                                         |
                                                                                        | (Audio Stream)
+---------------------------------------------------------------------------------------+-----------+
|                                     MICROSOFT EDGE TTS ENGINE                                     |
|  • Native Windows Async Event Loops -> Isolated Synchronous Request Wrapper                       |
|  • High-Fidelity Premium Voice Mappings (en-IN, hi-IN, kn-IN Neural Voices)                       |
+---------------------------------------------------------------------------------------------------+
```

## 🛠️ Installation & Setup

1. **Clone Repository**:
   ```bash
   git clone <repo-url>
   cd agribot_project
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the Application**:
   ```bash
   python manage.py runserver
   ```
   Navigate to `http://127.0.0.1:8000/` to launch the platform.

## 📖 Usage

* **Text Chat:** Simply type your agricultural queries. To trigger the ML models directly, provide the 7 critical parameters in your text (e.g., "N=90, P=42, K=43, temp=20.8, humidity=82, ph=6.5, rainfall=202").
* **Disease Detection:** Use the upload button to submit a photo of a diseased plant leaf. The ResNet50 model will analyze it and Gemini will provide treatment steps.
* **Voice Playback:** Ensure your browser allows audio playback. Select your preferred language (e.g., Kannada, Hindi) from the UI dropdown to hear the Neural TTS response.

## 👥 Team & Roles

* **VIVEK V RON:** ML Model Building | Backend Development | Team Lead
* **VIVEK TUBAKI:** Backend Development
* **VISHWANATH REDDY:** Frontend Development & Testing
* **VISHWAS HAVALADA:** Frontend Development & Testing

## 👨‍💻 Author

**Vivek V Ron** *Undergraduate Student, Computer Science and Engineering*
*MS Engineering College, Bengaluru*
