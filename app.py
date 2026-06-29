import os
import tempfile
import sys

import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PORT = int(os.environ.get("PORT", 8000))

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = Flask(__name__, template_folder="templates")
CORS(app)

recognizer = sr.Recognizer()

analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------
# SPEECH TO TEXT
# --------------------------------------------------

def run_wav_asr(audio_path):
    """
    Convert WAV audio into text using Google Speech Recognition.
    """

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source, duration=15)

        transcript = recognizer.recognize_google(audio)

        return transcript

    except sr.UnknownValueError:
        return "Speech was unintelligible."

    except Exception as e:
        return f"CRITICAL_ASR_ERROR: {str(e)}"


# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

def analyze_sentiment(text):

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    print("Sentiment Scores:", scores)

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
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

        # -----------------------------
        # Speech Recognition
        # -----------------------------

        transcript = run_wav_asr(temp_audio_path)

        if "CRITICAL_ASR_ERROR" in transcript:
            raise Exception(transcript)

        if "unintelligible" in transcript.lower():
            raise Exception(
                "Speech was not clear enough for transcription."
            )

        print("Transcript:", transcript)

        # -----------------------------
        # Sentiment Analysis
        # -----------------------------

        sentiment = analyze_sentiment(transcript)

        return jsonify({
            "transcript": transcript,
            "sentiment": sentiment,
            "status": "Complete"
        })

    except Exception as e:

        print(
            "Prediction Pipeline Error:",
            str(e),
            file=sys.stderr
        )

        return jsonify({
            "error": "Internal server error during analysis.",
            "details": str(e),
            "status": "Failed"
        }), 500

    finally:

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    print(f"Starting Flask on port {PORT}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
