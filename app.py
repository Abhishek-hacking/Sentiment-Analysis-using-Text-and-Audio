import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import speech_recognition as sr
from pydub import AudioSegment 
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import requests
import tempfile
import webbrowser
import threading
from scipy.io.wavfile import read as read_wav
import time

# --- CONFIGURATION ---
PORT = int(os.environ.get("PORT", 8000))

# CRITICAL FIX: Use the dedicated 3-Class RoBERTa model (Trained on Twitter data)
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# --- APP SETUP ---
app = Flask(__name__, template_folder='templates')
CORS(app) 
r = sr.Recognizer()

# --- LOAD MODELS ---
try:
    print("Loading models and encoders (This may take a moment)...")
    
except Exception as e:
    print(f"FATAL ERROR loading models: {e}", file=sys.stderr)
    sys.exit(1)


# --- FEATURE EXTRACTION UTILITIES (ASR) ---

def run_wav_asr(audio_path):
    """Transcribes pre-converted WAV audio using Google Web Speech API."""
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source, duration=15) 
            transcript = r.recognize_google(audio_data)
        return transcript
        
    except sr.UnknownValueError:
        return "Speech was unintelligible."
    except Exception as e:
        return f"CRITICAL_ASR_ERROR: {e}"


# --- FLASK ROUTES ---

@app.route('/')
def serve_index():
    """Serves the front-end HTML."""
    return render_template('index.html')


@app.route('/predict_sentiment', methods=['POST'])
def handle_prediction():
    """Handles the file upload, processes the audio, and returns the ensemble prediction."""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio_file']
    
    # Save file to a temporary location (Expects WAV format from frontend)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        temp_audio_path = tmp.name

    try:
        # 1. ASR Transcription
        transcript = run_wav_asr(temp_audio_path)
        
        if 'CRITICAL_ASR_ERROR' in transcript:
             raise Exception(f"ASR Failed: {transcript}")
        if 'unintelligible' in transcript:
             raise Exception("Speech was not clear enough for transcription.")

# Simple sentiment analysis using Hugging Face API

HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

response = requests.post(
    API_URL,
    headers=headers,
    json={"inputs": transcript}
)

result = response.json()

label = result[0][0]["label"]

if label == "POSITIVE":
    final_sentiment = "Positive"
elif label == "NEGATIVE":
    final_sentiment = "Negative"
else:
    final_sentiment = "Neutral"
        
        # --- Clean up and return ---
        return jsonify({
            'transcript': transcript,
            'sentiment': final_sentiment,
            'status': 'Complete'
        })

    except Exception as e:
        error_message = str(e)
        
        # Simplify error output for the frontend
        display_error = "Prediction failed due to unclear speech or model error."
        if "ASR Failed" in error_message:
            display_error = "ASR Failed: Check microphone/speak louder."
        elif "not clear enough" in error_message:
             display_error = "Please speak clearer: Audio was unintelligible."


        print(f"Prediction Pipeline Error: {e}", file=sys.stderr)
        return jsonify({'error': 'Internal server error during analysis.', 'details': display_error, 'status': 'Failed'}), 500
    
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == '__main__':
    import threading
    url = f"http://127.0.0.1:{PORT}"
    
    # Open browser automatically after a short delay (1 second)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    print(f"Starting Flask server on {url}. Opening browser now...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
