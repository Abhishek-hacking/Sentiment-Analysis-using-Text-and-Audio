import os
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import tempfile
import sys

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PORT = int(os.environ.get("PORT", 8000))

HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
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
    print("HF_TOKEN exists:", HF_TOKEN is not None)

    payload = {
        "inputs": text
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        print("Status Code:", response.status_code)
        print("Response:", response.text)

        response.raise_for_status()

        result = response.json()

        print(result)

        if isinstance(result, dict) and result.get("error"):
            raise Exception(result["error"])

        label = result[0]["label"]

        if label == "POSITIVE":
            return "Positive"
        elif label == "NEGATIVE":
            return "Negative"
        else:
            return "Neutral"

    except Exception as e:
        print("HF ERROR:", str(e))
        raise

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
