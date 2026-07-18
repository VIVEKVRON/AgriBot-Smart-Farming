import os
import re
import joblib
import pandas as pd
import numpy as np
import pickle
import sys # Added for Windows async fix
import tensorflow as tf
from PIL import Image

# NEW TTS IMPORTS
import edge_tts
import asyncio

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Tell Python to find and load the .env file
load_dotenv()

# Import your custom chatbot function
from .services.chatbot_service import ask_agribot

# ==========================================
# 1. LOAD MODELS GLOBALLY
# ==========================================
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models')
api_key = os.getenv("GEMINI_API_KEY")

# Pass it to Gemini
genai.configure(api_key=api_key)
llm = genai.GenerativeModel('gemini-2.5-flash')

# Initialize Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq: {e}")
    groq_client = None

def generate_ai_response(prompt, image=None):
    """Dual-LLM fallback routing function."""
    try:
        if image:
            return llm.generate_content([prompt, image]).text
        else:
            return llm.generate_content(prompt).text
    except Exception as gemini_err:
        print(f"DEBUG: Gemini API Failed: {str(gemini_err)}")
        if image:
            return "I am experiencing unusually high traffic and cannot analyze photos right now. Please try again in a few minutes."
        if groq_client:
            print("DEBUG: Falling back to Groq (Llama 3)...")
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                )
                return chat_completion.choices[0].message.content
            except Exception as groq_err:
                print(f"DEBUG: Groq API Failed: {str(groq_err)}")
                raise Exception(f"Both Gemini and Groq APIs failed. Gemini: {gemini_err}, Groq: {groq_err}")
        else:
            raise gemini_err


# Load Crop Models
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_crop_model.pkl'))
crop_encoder = joblib.load(os.path.join(MODEL_DIR, 'crop_label_encoder.pkl'))

# Load Disease Models
disease_model = load_model(os.path.join(MODEL_DIR, 'best_resnet_plant_model.h5'))
disease_classes = joblib.load(os.path.join(MODEL_DIR, 'plant_disease_classes.pkl'))

# Load Yield Prediction Models
rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_yield_model.pkl'))
yield_columns = joblib.load(os.path.join(MODEL_DIR, 'yield_model_columns.pkl'))

# Load Fertilizer Model
try:
    fert_model_path = os.path.join(MODEL_DIR, 'fertilizer_model.pkl')
    fertilizer_model = pickle.load(open(fert_model_path, 'rb'))
except Exception as e:
    print(f"Error loading fertilizer model: {e}")
    fertilizer_model = None

# Load FAISS & NLP Router
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = os.path.join(MODEL_DIR, 'faiss_agri_index')
    vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS Knowledge Base loaded")
except Exception as e:
    print(f"❌ FAISS Load Error: {e}")

try:
    nlp_path = os.path.join(MODEL_DIR, 'nlp_intent_model.pkl')
    nlp_router = pickle.load(open(nlp_path, 'rb'))
    print("✅ NLP Router loaded")
except Exception as e:
    print(f"❌ NLP Router Load Error: {e}")


# ==========================================
# 2. STANDARD VIEWS
# ==========================================

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
    """Handles plant image uploads for the standalone HTML page."""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            img_path = fs.path(filename)
            uploaded_file_url = fs.url(filename)
            
            # Multimodal Security Gatekeeper
            gatekeeper_img = Image.open(img_path)
            gatekeeper_prompt = 'Is this an image of a plant, leaf, crop, or agriculture? Answer strictly with a single word: YES or NO.'
            gatekeeper_response = generate_ai_response(gatekeeper_prompt, image=gatekeeper_img)
            if 'NO' in gatekeeper_response.upper():
                return render(request, 'disease_predict.html', {'error': 'Please upload a valid image of a plant or leaf.'})
            
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
            
            return render(request, 'disease_predict.html', {
                'result': clean_disease_name,
                'uploaded_image': uploaded_file_url
            })
        except Exception as e:
            return render(request, 'disease_predict.html', {'error': f"Error: {str(e)}"})

    return render(request, 'disease_predict.html')

def predict_yield(request):
    """Handles the crop yield prediction using the Random Forest model."""
    if request.method == 'POST':
        try:
            year = int(request.POST.get('Year'))
            rainfall = float(request.POST.get('average_rain_fall_mm_per_year'))
            pesticides = float(request.POST.get('pesticides_tonnes'))
            temp = float(request.POST.get('avg_temp'))
            area = request.POST.get('Area')
            item = request.POST.get('Item')

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
                input_dict = {col: 0 for col in yield_columns}

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

                area_col = f'Area_{area}'
                item_col = f'Item_{item}'
                if area_col in input_dict: 
                    input_dict[area_col] = 1
                if item_col in input_dict: 
                    input_dict[item_col] = 1

                input_df = pd.DataFrame([input_dict])
                input_df = input_df[list(yield_columns)]

            predicted_yield = rf_model.predict(input_df)[0]
            result_text = f"{round(predicted_yield, 2)} hg/ha"

            return render(request, 'yield_predict.html', {'result': result_text})
            
        except Exception as e:
            expected_cols = list(yield_columns)[:8]
            return render(request, 'yield_predict.html', {'error': f"Error: {str(e)} | First few expected columns are: {expected_cols}"})

    return render(request, 'yield_predict.html')

def predict_fertilizer(request):
    """Handles Fertilizer recommendations."""
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
                
                explanation = ""
                if '-' in result_str and len(result_str.split('-')) == 3:
                    n_val, p_val, k_val = result_str.split('-')
                    explanation = f"This is an NPK fertilizer blend containing <strong>{n_val}% Nitrogen</strong> (for leaf growth), <strong>{p_val}% Phosphorus</strong> (for strong roots and fruiting), and <strong>{k_val}% Potassium</strong> (for overall plant health and disease resistance)."
                elif "urea" in result_str.lower():
                    explanation = "Urea is a highly concentrated nitrogen fertilizer (46-0-0) used to promote rapid, green leafy growth."
                elif "dap" in result_str.lower():
                    explanation = "DAP (Diammonium Phosphate) provides a massive boost of Phosphorus for root development, along with a starting dose of Nitrogen."
                elif "14-35-14" in result_str:
                    explanation = "This blend is heavily focused on Phosphorus (35%), making it ideal for the early stages of root establishment and seed development."
                else:
                    explanation = f"This fertilizer is highly recommended based on your soil's current nutrient deficit and local weather conditions."
                
                context['explanation'] = explanation
            else:
                context['error'] = "Model file not found. Please contact the administrator."
                
        except Exception as e:
            context['error'] = f"An error occurred during prediction: {str(e)}"
            
    return render(request, 'fertilizer_predict.html', context)


# ==========================================
# 3. CHATBOT AND ML API ENDPOINTS
# ==========================================

def extract_agricultural_parameters(text):
    """
    DETERMINISTICALLY detects agricultural parameters in casual text.
    Returns a dictionary of whatever it finds (can be partial).
    """
    try:
        patterns = {
            'N': r"\b(?:nitrogen|N)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)",
            'P': r"\b(?:phosphorus|P)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)",
            'K': r"\b(?:potassium|K)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)",
            'ph': r"\b(?:ph)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)",
            'temperature': r"\b(?:temperature|temp|T)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)",
            'humidity': r"\b(?:humidity|hum|H)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)",
            'rainfall': r"\b(?:rainfall|rain|R)\b\s*[:=\s]\s*(\d+(?:\.\d+)?)"
        }

        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted[key] = float(match.group(1))
                
        return extracted if extracted else None
        
    except Exception as e:
        print(f"DEBUG: Error extracting parameters: {e}")
        return None
    
def validate_agri_parameters(params):
    """
    Validates the extracted parameters to ensure they fall within realistic 
    biological and chemical boundaries. Null-safe for partial data.
    Returns a tuple: (is_valid: bool, error_message: str)
    """
    bounds = {
        'N': (0, 500, "Nitrogen (N)"),
        'P': (0, 500, "Phosphorus (P)"),
        'K': (0, 500, "Potassium (K)"),
        'temperature': (-10, 60, "Temperature (°C)"),
        'humidity': (0, 100, "Humidity (%)"),
        'ph': (0, 14, "pH"),
        'rainfall': (0, 1000, "Rainfall (mm)")
    }
    
    for key, (min_val, max_val, name) in bounds.items():
        val = params.get(key)
        # ONLY validate if the parameter actually exists in the dictionary
        if val is not None:
            if not (min_val <= val <= max_val):
                return False, f"I noticed your {name} value ({val}) is outside realistic agricultural limits ({min_val} to {max_val}). Please double-check your data!"
    
    return True, ""

def chat_api(request):
    """Handles text messages using the NLP + RAG Pipeline and forced ML paths with Memory."""
    if request.method == 'POST':
        user_text = request.POST.get('message', '').strip()
        # Fetch the language code if your frontend sends it, otherwise default to English
        lang_code = request.POST.get('lang', 'en-IN') 
        
        # Map the code to the actual language name for the Gemini prompt
        lang_map = {'en-IN': 'English', 'hi-IN': 'Hindi', 'kn-IN': 'Kannada'}
        target_lang = lang_map.get(lang_code, 'the exact same language as the user')

        if not user_text:
            return JsonResponse({'error': 'No message provided.'}, status=400)

        # --- Step A: DETERMINISTIC PRE-ROUTER CHECK & PARAMETER ACCUMULATION ---
        new_params = extract_agricultural_parameters(user_text)
        
        # 1. Fetch previously accumulated parameters from the session (or start fresh)
        accumulated_params = request.session.get('accumulated_params', {})
        
        # 2. If new parameters are found, update the session memory
        if new_params:
            accumulated_params.update(new_params)
            request.session['accumulated_params'] = accumulated_params
            print(f"DEBUG: Current Accumulated Params: {accumulated_params}")

        required_keys = {'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'}
        
        # 3. Check if we have SOME parameters, but not ALL
        if accumulated_params and not required_keys.issubset(accumulated_params.keys()):
            missing_keys = list(required_keys - set(accumulated_params.keys()))
            missing_str = ", ".join([k.upper() if len(k)==1 else k.title() for k in missing_keys])
            
            return JsonResponse({
                'response': f"Got it! I have saved your data. To run the exact mathematical calculation, I still need your: <strong>{missing_str}</strong>.",
                'missing_parameters': missing_keys  # NEW: Send the exact missing keys to the frontend UI
            })

        # 4. If we have ALL 7 parameters, proceed to the ML Model
        if accumulated_params and required_keys.issubset(accumulated_params.keys()):
            agri_params = accumulated_params
            print(f"DEBUG: All 7 Parameters Acquired. Forcing ML Path: {agri_params}")
            
            # --- SECURITY GUARDRAIL (VALIDATION) ---
            is_valid, error_message = validate_agri_parameters(agri_params)
            if not is_valid:
                print(f"DEBUG: Validation Failed - {error_message}")
                # Clear the bad parameter so they can try again without getting stuck in a loop
                request.session.pop('accumulated_params', None)
                return JsonResponse({'response': error_message})
            # ------------------------------

            # Save to standard params for Fertilizer follow-up, then clear accumulation tracker
            request.session['saved_params'] = agri_params
            request.session.pop('accumulated_params', None)
            
            try:
                # 1. Execute XGBoost Mathematical Calculation safely
                input_data = pd.DataFrame([[
                    agri_params['N'], agri_params['P'], agri_params['K'], 
                    agri_params['temperature'], agri_params['humidity'], 
                    agri_params['ph'], agri_params['rainfall']
                ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
                
                prediction_encoded = xgb_model.predict(input_data)
                recommended_crop = crop_encoder.inverse_transform(prediction_encoded)[0]
                request.session['saved_crop'] = recommended_crop
                
                analytical_prompt = f"""
                    The user has provided exact agricultural parameters in the chat:
                    (N={agri_params['N']}, P={agri_params['P']}, K={agri_params['K']}, pH={agri_params['ph']}, 
                     Temperature={agri_params['temperature']}°C, Humidity={agri_params['humidity']}%, Rainfall={agri_params['rainfall']}mm).

                    Based strictly on this numerical input, our mathematical XGBoost model has calculated 
                    the optimal crop to grow is: {recommended_crop.upper()}.

                    INSTRUCTIONS:
                    1. Respond conversationally and politely. 
                    2. Confirm the numbers they provided and state the model's recommendation. Mention it's a mathematical calculation.
                    3. Suggest they can also ask about fertilizer for this crop.
                    4. CRITICAL: You MUST write your entire response in {target_lang.upper()} (or match the language of the user's input).
                """
                
                # 2. Layered Protection for the LLM API
                try:
                    gemini_response = generate_ai_response(analytical_prompt)
                except Exception as api_error:
                    print(f"DEBUG: Gemini API Failed (Likely Rate Limit) - {str(api_error)}")
                    # FALLBACK: Bypass Gemini and return the raw ML calculation
                    gemini_response = f"Based on the exact soil and weather parameters you provided, my mathematical XGBoost model recommends growing: <strong>{recommended_crop.upper()}</strong>."
                
                return JsonResponse({'response': gemini_response})
                
            except Exception as e:
                # 3. Expose the EXACT Python error to the chat interface if Pandas or XGBoost crashes
                print(f"DEBUG: Critical Error in ML prediction path: {str(e)}")
                return JsonResponse({'response': f"Sorry, I encountered an internal calculation error: <strong>{str(e)}</strong>"})

        # --- Step B: MEMORY CHECK FOR FERTILIZER FOLLOW-UP ---
        saved_params = request.session.get('saved_params')
        saved_crop = request.session.get('saved_crop')
        
        follow_up_keywords = ['yes', 'yeah', 'sure', 'fertilizer', 'recommendation', 'please', 'ok', 'ಹೌದು', 'ಗೊಬ್ಬರ', 'हाँ', 'उर्वरक', 'खाद']
        is_follow_up = any(word in user_text.lower() for word in follow_up_keywords)
        
        # New check specifically for fertilizer intent
        is_asking_fertilizer = any(word in user_text.lower() for word in ['fertilizer', 'ಗೊಬ್ಬರ', 'उर्वरक', 'खाद'])

        if saved_params and saved_crop and is_follow_up:
            print("DEBUG: Follow-up detected! Using saved session memory for Fertilizer ML.")
            try:
                if fertilizer_model:
                    input_features = np.array([[
                        saved_params['N'], saved_params['P'], saved_params['K'], 
                        saved_params['temperature'], saved_params['humidity'], 
                        saved_params['ph'], saved_params['rainfall']
                    ]])
                    
                    prediction = fertilizer_model.predict(input_features)
                    fert_result = str(prediction[0])
                    
                    fert_prompt = f"""
                        The user previously asked for a crop recommendation, and we calculated {saved_crop.upper()}.
                        Now they are asking for a fertilizer recommendation for that exact same field.
                        The saved soil parameters are: N={saved_params['N']}, P={saved_params['P']}, K={saved_params['K']}, pH={saved_params['ph']}.
                        
                        Our mathematical Fertilizer ML model recommends exactly this fertilizer: {fert_result}.
                        
                        INSTRUCTIONS:
                        1. Respond conversationally. State the recommended fertilizer ({fert_result}) and explain briefly why it is appropriate for growing {saved_crop.upper()}. 
                        2. Do not ask them for parameters again.
                        3. CRITICAL: You MUST write your entire response in {target_lang.upper()} (or match the language of the user's input).
                    """
                    gemini_response = generate_ai_response(fert_prompt)
                    
                    request.session.pop('saved_params', None)
                    request.session.pop('saved_crop', None)
                    
                    return JsonResponse({'response': gemini_response})
                else:
                    return JsonResponse({'response': "I'm sorry, my fertilizer model is currently unavailable."})
                    
            except Exception as e:
                print(f"DEBUG: Fertilizer memory error: {e}")
                
        # --- SECURITY GUARDRAIL (SESSION AMNESIA INTERCEPT) ---
        elif is_asking_fertilizer and not saved_params:
            print("DEBUG: Fertilizer requested but session memory is empty.")
            return JsonResponse({
                'response': "It looks like we are starting a fresh session! To calculate the exact fertilizer mathematically, I need your field's current parameters (N, P, K, pH, Temperature, Humidity, and Rainfall). Could you provide those numbers?"
            })
        # ----------------------------------------------------------

        # --- Step C: FALLBACK CONVERSATIONAL PATH (RAG) ---
        print("DEBUG: General Query. Proceeding with Conversational RAG Path.")
        try:
            intent = nlp_router.predict([user_text])[0]
            docs = vector_db.similarity_search(user_text, k=2)
            pdf_context = " ".join([d.page_content for d in docs])

            prompt = f"""
            SYSTEM ROLE: You are AgriBot, an expert agricultural assistant.
            DETECTED INTENT: {intent}
            FACTUAL EVIDENCE: {pdf_context}
            FARMER'S QUESTION: ### {user_text} ###
            """

            final_response = generate_ai_response(prompt)
            return JsonResponse({'response': final_response})
            
        except Exception as e:
            # FIX: Print the ACTUAL error to the chat instead of a generic message
            error_msg = f"RAG/LLM Error: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return JsonResponse({'response': error_msg})
            
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def detect_disease(request):
    """Handles plant image uploads using the ResNet CNN model with threshold guardrails."""
    if request.method == 'POST' and request.FILES.get('plant_image'):
        try:
            uploaded_file = request.FILES['plant_image']
            
            img = Image.open(uploaded_file).convert('RGB')
            
            # Multimodal Security Gatekeeper
            gatekeeper_prompt = 'Is this an image of a plant, leaf, crop, or agriculture? Answer strictly with a single word: YES or NO.'
            gatekeeper_response = generate_ai_response(gatekeeper_prompt, image=img)
            if 'NO' in gatekeeper_response.upper():
                return JsonResponse({
                    "status": "success",
                    "disease": "Invalid Image",
                    "bot_response": "This doesn't look like a plant to me! Please upload a clear picture of a crop or leaf."
                })

            img = img.resize((224, 224)) 
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            predictions = disease_model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # --- SECURITY GUARDRAIL (CONFIDENCE THRESHOLD) ---
            if confidence < 65.0:
                print(f"DEBUG: Image rejected due to low confidence ({confidence:.2f}%)")
                return JsonResponse({
                    "status": "success",
                    "disease": "Unrecognized Image",
                    "confidence": f"{confidence:.2f}%",
                    "bot_response": "I am having trouble recognizing this image. It does not appear to be a clear crop leaf, or the disease is too ambiguous. Please upload a clear, well-lit photo of the affected plant."
                })
            # -----------------------------------------------------
            
            predicted_disease = disease_classes[predicted_class_index]
            clean_disease_name = predicted_disease.replace('___', ' - ').replace('_', ' ')
            
            if "healthy" in clean_disease_name.lower():
                return JsonResponse({
                    "status": "success",
                    "disease": clean_disease_name,
                    "confidence": f"{confidence:.2f}%",
                    "bot_response": "Great news! Your plant looks healthy. Keep up your current routine."
                })
                
            prompt = f"""
            SYSTEM ROLE: You are an expert agricultural AI.
            DIAGNOSIS: {clean_disease_name}
            
            INSTRUCTIONS: Write a short, empathetic response explaining what this disease is and provide 2-3 actionable steps the farmer can take to cure it.
            """
            llm_response = generate_ai_response(prompt)
            
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

# ==========================================
# 4. TEXT-TO-SPEECH API (edge-tts)
# ==========================================

# CRITICAL FIX FOR WINDOWS ASYNCIO BUGS:
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def generate_audio_sync(text, voice, rate):
    """Runs the async edge-tts code safely in an isolated, synchronous Django thread."""
    async def _amain():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    
    # Create a fresh, isolated event loop for this specific request so Django doesn't crash
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_amain())
    finally:
        loop.close()

def speak_text(request):
    """
    Converts text to speech using premium Microsoft Edge Neural voices.
    Allows for gender selection and native playback speed adjustments.
    """
    text = request.GET.get('text', '')
    lang_code = request.GET.get('lang', 'en-IN')
    gender = request.GET.get('gender', 'female').lower() # Default to female
    
    # Natively speed up the audio! "+50%" equals roughly 1.5x speed.
    speed_multiplier = request.GET.get('speed', '+50%') 

    # Map frontend codes to Microsoft's premium Neural voices
    voice_map = {
        'en-IN': {
            'female': 'en-IN-NeerjaNeural',
            'male': 'en-IN-PrabhatNeural'
        },
        'hi-IN': {
            'female': 'hi-IN-SwaraNeural',
            'male': 'hi-IN-MadhurNeural'
        },
        'kn-IN': {
            'female': 'kn-IN-SapnaNeural',
            'male': 'kn-IN-GaganNeural'
        }
    }

    try:
        selected_voice = voice_map[lang_code][gender]
    except KeyError:
        selected_voice = 'en-IN-NeerjaNeural' # Fallback

    if text:
        try:
            # Use our robust synchronous wrapper instead of asgiref
            audio_bytes = generate_audio_sync(text, selected_voice, speed_multiplier)
            return HttpResponse(audio_bytes, content_type="audio/mpeg")
        except Exception as e:
            # Print the EXACT error to the terminal if it fails
            print(f"DEBUG: TTS Error - {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'No text provided'}, status=400)