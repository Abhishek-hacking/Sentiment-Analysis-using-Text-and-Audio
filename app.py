import os
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import tempfile
import sys

from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"]
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PORT = int(os.environ.get("PORT", 8000))

HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = Flask(__name__, template_folder="templates")
CORS(app)

recognizer = sr.Recognizer()

# --------------------------------------------------
# SPEECH TO TEXT
# --------------------------------------------------

def run_wav_asr(audio_path):
    """
    Convert WAV audio into text using Google Speech Recognition.
    """
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source, duration=15)
            transcript = recognizer.recognize_google(audio_data)

        return transcript

    except sr.UnknownValueError:
        return "Speech was unintelligible."

    except Exception as e:
        return f"CRITICAL_ASR_ERROR: {str(e)}"


# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

def analyze_sentiment(text):
    result = client.text_classification(
        text,
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    label = result[0]["label"]

    if label == "POSITIVE":
        return "Positive"
    elif label == "NEGATIVE":
        return "Negative"
    return "Neutral"

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def serve_index():
    return render_template("index.html")


@app.route("/predict_sentiment", methods=["POST"])
def handle_prediction():

    if "audio_file" not in request.files:
        return jsonify({
            "error": "No audio file provided."
        }), 400

    audio_file = request.files["audio_file"]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        audio_file.save(tmp.name)
        temp_audio_path = tmp.name

    try:

        # ------------------------------------------
        # Speech Recognition
        # ------------------------------------------

        transcript = run_wav_asr(temp_audio_path)

        if "CRITICAL_ASR_ERROR" in transcript:
            raise Exception(transcript)

        if "unintelligible" in transcript.lower():
            raise Exception(
                "Speech was not clear enough for transcription."
            )

        # ------------------------------------------
        # Sentiment Analysis
        # ------------------------------------------

        final_sentiment = analyze_sentiment(transcript)

        return jsonify({
            "transcript": transcript,
            "sentiment": final_sentiment,
            "status": "Complete"
        })

    except Exception as e:

        error_message = str(e)

        display_error = (
            "Prediction failed due to unclear speech "
            "or sentiment model error."
        )

        if "ASR" in error_message:
            display_error = (
                "ASR Failed: Check microphone "
                "or speak louder."
            )

        elif "clear enough" in error_message:
            display_error = (
                "Please speak more clearly."
            )

        print(
            f"Prediction Pipeline Error: {error_message}",
            file=sys.stderr
        )

        return jsonify({
            "error": "Internal server error during analysis.",
            "details": display_error,
            "status": "Failed"
        }), 500

    finally:

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route("/test-google")
def test_google():
    try:
        r = requests.get("https://www.google.com", timeout=10)
        return {
            "status": r.status_code,
            "message": "Google reachable"
        }
    except Exception as e:
        return {
            "error": str(e)
        }, 500


@app.route("/test-hf")
def test_hf():
    try:
        r = requests.get(
            "https://api-inference.huggingface.co",
            timeout=10
        )
        return {
            "status": r.status_code,
            "text": r.text[:200]
        }
    except Exception as e:
        return {
            "error": str(e)
        }, 500


@app.route("/test-dns")
def test_dns():
    import socket

    try:
        return {
            "google": socket.gethostbyname("google.com"),
            "hf": socket.gethostbyname("api-inference.huggingface.co")
        }
    except Exception as e:
        return {"error": str(e)}, 500



# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Starting Flask server on port {PORT}")
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
