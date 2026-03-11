import os
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from django.core.files.storage import FileSystemStorage

import pickle
import tensorflow as tf
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

import os
from dotenv import load_dotenv

# Tell Python to find and load the .env file
load_dotenv()

# Import your custom chatbot function
from .services.chatbot_service import ask_agribot

# 1. Define paths to your ML models
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models')
# Fetch the key from the .env file securely
api_key = os.getenv("GEMINI_API_KEY")

# Pass it to Gemini
genai.configure(api_key=api_key)
llm = genai.GenerativeModel('gemini-2.5-flash')

# 2. Load models globally (so they only load once when the server starts)
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_crop_model.pkl'))
crop_encoder = joblib.load(os.path.join(MODEL_DIR, 'crop_label_encoder.pkl'))
disease_model = load_model(os.path.join(MODEL_DIR, 'best_resnet_plant_model.h5'))
disease_classes = joblib.load(os.path.join(MODEL_DIR, 'plant_disease_classes.pkl'))

# Load Yield Prediction Models
rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_yield_model.pkl'))
yield_columns = joblib.load(os.path.join(MODEL_DIR, 'yield_model_columns.pkl'))
# --- VIEWS ---

def home(request):
    """Renders the main homepage."""
    return render(request, 'index.html')

def predict_crop(request):
    """Handles the crop recommendation form and XGBoost prediction."""
    if request.method == 'POST':
        try:
            n = float(request.POST.get('nitrogen'))
            p = float(request.POST.get('phosphorous'))
            k = float(request.POST.get('potassium'))
            temp = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            ph = float(request.POST.get('ph'))
            rainfall = float(request.POST.get('rainfall'))

            input_data = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], 
                                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            prediction_encoded = xgb_model.predict(input_data)
            recommended_crop = crop_encoder.inverse_transform(prediction_encoded)[0]

            return render(request, 'crop_predict.html', {'result': recommended_crop.upper()})
        except Exception as e:
            return render(request, 'crop_predict.html', {'error': f"Error: {str(e)}"})

    return render(request, 'crop_predict.html')

def predict_disease(request):
    """Handles plant image uploads and ResNet50 predictions."""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            img_path = fs.path(filename)
            uploaded_file_url = fs.url(filename)
            
            # Preprocess image for ResNet50
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            predictions = disease_model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_disease = disease_classes[predicted_class_index]
            clean_disease_name = predicted_disease.replace('___', ' - ').replace('_', ' ')
            
            #os.remove(img_path) # Clean up temporary image
            
            return render(request, 'disease_predict.html', {
                'result': clean_disease_name,
                'uploaded_image': uploaded_file_url
            })
        except Exception as e:
            return render(request, 'disease_predict.html', {'error': f"Error: {str(e)}"})

    return render(request, 'disease_predict.html')
    
# --- GLOBALLY LOAD AI COMPONENTS FOR THE CHATBOT ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Load Knowledge Base (FAISS)
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = os.path.join(base_dir, 'faiss_agri_index')
    vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"FAISS Load Error: {e}")

# 3. Load NLP Router
try:
    nlp_path = os.path.join(base_dir, 'nlp_intent_model.pkl')
    nlp_router = pickle.load(open(nlp_path, 'rb'))
except Exception as e:
    print(f"NLP Router Load Error: {e}")

# 4. Load ResNet Model for Disease
try:
    resnet_path = os.path.join(base_dir, 'best_resnet_plant_model.h5')
    disease_model = tf.keras.models.load_model(resnet_path)
except Exception as e:
    print(f"ResNet Load Error: {e}")

def chat_api(request):
    """Handles text messages using the NLP + RAG Pipeline."""
    if request.method == 'POST':
        user_text = request.POST.get('message')
        if not user_text:
            return JsonResponse({'error': 'No message provided.'}, status=400)

        try:
            # 1. NLP Intent Classification
            intent = nlp_router.predict([user_text])[0]

            # 2. FAISS Retrieval
            docs = vector_db.similarity_search(user_text, k=2)
            pdf_context = " ".join([d.page_content for d in docs])

            # 3. Gemini Synthesis
            prompt = f"""
            SYSTEM ROLE: You are AgriBot, an expert agricultural assistant.
            DETECTED INTENT: {intent}
            FACTUAL EVIDENCE: {pdf_context}

            FARMER'S QUESTION: "{user_text}"

            INSTRUCTIONS: 
            1. Answer the farmer naturally and politely.
            2. Use the FACTUAL EVIDENCE to ground your answer. Do not guess or hallucinate.
            3. If the user asks for a specific numerical prediction (like exact fertilizer amounts or crop yield), politely tell them to use the dedicated prediction tools in the top navigation bar.
            """

            final_response = llm.generate_content(prompt).text
            return JsonResponse({'response': final_response})
            
        except Exception as e:
            print(f"Chatbot RAG error: {e}")
            return JsonResponse({'response': "I am currently experiencing technical difficulties processing your request. Please try again."})
            
    return JsonResponse({'error': 'Invalid request method.'}, status=405)
    
def predict_yield(request):
    """Handles the crop yield prediction using the Random Forest model."""
    if request.method == 'POST':
        try:
            # 1. Grab numerical and categorical inputs from the form
            year = int(request.POST.get('Year'))
            rainfall = float(request.POST.get('average_rain_fall_mm_per_year'))
            pesticides = float(request.POST.get('pesticides_tonnes'))
            temp = float(request.POST.get('avg_temp'))
            area = request.POST.get('Area')
            item = request.POST.get('Item')

            # 2. Check if the model is a Pipeline that expects raw unencoded data
            # (Some Kaggle models handle the encoding automatically)
            if 'Area' in yield_columns and 'Item' in yield_columns:
                input_df = pd.DataFrame([{
                    'Area': area,
                    'Item': item,
                    'Year': year,
                    'average_rain_fall_mm_per_year': rainfall,
                    'pesticides_tonnes': pesticides,
                    'avg_temp': temp
                }])
            else:
                # 3. Otherwise, do manual one-hot encoding dynamically
                input_dict = {col: 0 for col in yield_columns}

                # Dynamically match numerical columns (ignores capitalization)
                for col in yield_columns:
                    col_lower = col.lower()
                    if 'year' in col_lower: 
                        input_dict[col] = year
                    elif 'rain' in col_lower: 
                        input_dict[col] = rainfall
                    elif 'pest' in col_lower: 
                        input_dict[col] = pesticides
                    elif 'temp' in col_lower: 
                        input_dict[col] = temp

                # Set specific one-hot encoded categorical columns to 1
                area_col = f'Area_{area}'
                item_col = f'Item_{item}'
                if area_col in input_dict: 
                    input_dict[area_col] = 1
                if item_col in input_dict: 
                    input_dict[item_col] = 1

                # Convert to DataFrame and ENFORCE exact column order
                input_df = pd.DataFrame([input_dict])
                input_df = input_df[list(yield_columns)]

            # 4. Make the prediction
            predicted_yield = rf_model.predict(input_df)[0]
            
            # 5. Format the result
            result_text = f"{round(predicted_yield, 2)} hg/ha"

            return render(request, 'yield_predict.html', {'result': result_text})
            
        except Exception as e:
            # If it still fails, print the exact expected columns to the screen for easy debugging!
            expected_cols = list(yield_columns)[:8]
            return render(request, 'yield_predict.html', {'error': f"Error: {str(e)} | First few expected columns are: {expected_cols}"})

    return render(request, 'yield_predict.html')

# --- GLOBALLY LOAD AI COMPONENTS FOR THE CHATBOT ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_models_dir = os.path.join(base_dir, 'ml_models') # Pointing to the correct folder!

# 2. Load Knowledge Base (FAISS)
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = os.path.join(ml_models_dir, 'faiss_agri_index')
    vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS Knowledge Base loaded")
except Exception as e:
    print(f"❌ FAISS Load Error: {e}")

# 3. Load NLP Router
try:
    nlp_path = os.path.join(ml_models_dir, 'nlp_intent_model.pkl')
    nlp_router = pickle.load(open(nlp_path, 'rb'))
    print("✅ NLP Router loaded")
except Exception as e:
    print(f"❌ NLP Router Load Error: {e}")

# 4. Load ResNet Model for Disease
try:
    resnet_path = os.path.join(ml_models_dir, 'best_resnet_plant_model.h5')
    disease_model = tf.keras.models.load_model(resnet_path)
    print("✅ ResNet Disease Model loaded")
except Exception as e:
    print(f"❌ ResNet Load Error: {e}")

# ==========================================
# 2. THE DJANGO VIEW (API Endpoint)
# ==========================================
#@csrf_exempt  # Use CSRF tokens properly in your production frontend
def detect_disease(request):
    """Handles plant image uploads using the ResNet CNN model."""
    if request.method == 'POST' and request.FILES.get('plant_image'):
        try:
            uploaded_file = request.FILES['plant_image']
            
            # Preprocess the image for ResNet
            img = Image.open(uploaded_file).convert('RGB')
            img = img.resize((224, 224)) 
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            # CNN Prediction
            predictions = disease_model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            predicted_disease = disease_classes[predicted_class_index]
            clean_disease_name = predicted_disease.replace('___', ' - ').replace('_', ' ')
            
            if "healthy" in clean_disease_name.lower():
                return JsonResponse({
                    "status": "success",
                    "disease": clean_disease_name,
                    "confidence": f"{confidence:.2f}%",
                    "bot_response": "Great news! Your plant looks healthy. Keep up your current routine."
                })
                
            # RAG Synthesis for Diseased Plants
            prompt = f"""
            SYSTEM ROLE: You are an expert agricultural AI.
            DIAGNOSIS: {clean_disease_name}
            
            INSTRUCTIONS: Write a short, empathetic response explaining what this disease is and provide 2-3 actionable steps the farmer can take to cure it.
            """
            llm_response = llm.generate_content(prompt).text
            
            return JsonResponse({
                "status": "success",
                "disease": clean_disease_name,
                "confidence": f"{confidence:.2f}%",
                "bot_response": llm_response
            })
            
        except Exception as e:
            print(f"Disease detection error: {e}")
            return JsonResponse({"status": "error", "message": "Failed to analyze image."}, status=500)
            
    return JsonResponse({"status": "error", "message": "No image uploaded."}, status=400)

import pickle
import numpy as np
import os
from django.shortcuts import render

# Load the model once at the top of your file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fert_model_path = os.path.join(base_dir, 'ml_models', 'fertilizer_model.pkl')

try:
    fertilizer_model = pickle.load(open(fert_model_path, 'rb'))
except Exception as e:
    print(f"Error loading fertilizer model: {e}")
    fertilizer_model = None

def predict_fertilizer(request):
    context = {}
    
    if request.method == 'POST':
        try:
            n = float(request.POST.get('nitrogen'))
            p = float(request.POST.get('phosphorus'))
            k = float(request.POST.get('potassium'))
            temp = float(request.POST.get('temperature'))
            hum = float(request.POST.get('humidity'))
            ph = float(request.POST.get('ph'))
            rainfall = float(request.POST.get('rainfall'))
            
            input_features = np.array([[n, p, k, temp, hum, ph, rainfall]])
            
            if fertilizer_model:
                prediction = fertilizer_model.predict(input_features)
                result_str = str(prediction[0])
                context['result'] = result_str
                
                # --- NEW: Dynamic Explanation Logic ---
                explanation = ""
                # Check if the result is an NPK ratio (e.g., "10-26-26")
                if '-' in result_str and len(result_str.split('-')) == 3:
                    n_val, p_val, k_val = result_str.split('-')
                    explanation = f"This is an NPK fertilizer blend containing <strong>{n_val}% Nitrogen</strong> (for leaf growth), <strong>{p_val}% Phosphorus</strong> (for strong roots and fruiting), and <strong>{k_val}% Potassium</strong> (for overall plant health and disease resistance)."
                
                # Check for common name-based fertilizers
                elif "urea" in result_str.lower():
                    explanation = "Urea is a highly concentrated nitrogen fertilizer (46-0-0) used to promote rapid, green leafy growth."
                elif "dap" in result_str.lower():
                    explanation = "DAP (Diammonium Phosphate) provides a massive boost of Phosphorus for root development, along with a starting dose of Nitrogen."
                elif "14-35-14" in result_str:
                    explanation = "This blend is heavily focused on Phosphorus (35%), making it ideal for the early stages of root establishment and seed development."
                else:
                    explanation = f"This fertilizer is highly recommended based on your soil's current nutrient deficit and local weather conditions."
                
                context['explanation'] = explanation
                # --------------------------------------

            else:
                context['error'] = "Model file not found. Please contact the administrator."
                
        except Exception as e:
            context['error'] = f"An error occurred during prediction: {str(e)}"
            
    return render(request, 'fertilizer_predict.html', context)