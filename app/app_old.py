from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import re

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "..", "web")
BUSINESS_FILE = os.path.join(BASE_DIR, "business.txt")

# ---------------------------------------------------
# Flask
# ---------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------
# Ollama Settings
# ---------------------------------------------------
OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL_NAME = "qwen2.5:3b-instruct"

# ---------------------------------------------------
# Global Cache
# ---------------------------------------------------
LAST_MODIFIED = 0
CHUNKS = []

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def split_chunks(text):
    """
    Split by blank lines first.
    Fallback to line-by-line if needed.
    """
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]

    return parts


def tokenize(text):
    """
    Lowercase alphanumeric words only.
    """
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def load_business_data():
    """
    Reload business.txt only when changed.
    """
    global LAST_MODIFIED, CHUNKS

    if not os.path.exists(BUSINESS_FILE):
        CHUNKS = []
        return

    mtime = os.path.getmtime(BUSINESS_FILE)

    if mtime != LAST_MODIFIED:
        with open(BUSINESS_FILE, "r", encoding="utf-8") as f:
            raw_text = f.read()

        CHUNKS = split_chunks(raw_text)
        LAST_MODIFIED = mtime


def retrieve_chunks(question, top_n=3):
    """
    Keyword scoring retrieval.
    """
    q_words = tokenize(question)
    scored = []

    for chunk in CHUNKS:
        c_words = tokenize(chunk)
        score = len(q_words.intersection(c_words))

        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        return CHUNKS[:top_n]

    return [chunk for score, chunk in scored[:top_n]]


def is_greeting(text):
    greetings = {
        "hi", "hello", "hey", "good morning",
        "good afternoon", "good evening"
    }
    return text.lower().strip() in greetings


# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        load_business_data()

        data = request.json or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please enter a message."})

        # Fast path greeting
        if is_greeting(user_message):
            return jsonify({"reply": "Hello! How can I help you today?"})

        # Retrieve relevant business info
        relevant_chunks = retrieve_chunks(user_message, top_n=3)
        context = "\n\n".join(relevant_chunks)

        prompt = f"""
You are a business customer support assistant.

Use ONLY the information below.

Business Information:
{context}

Strict Rules:
- Maximum 40 words.
- Clear and friendly.
- Answer only the user's question.
- Do not add extra details unless asked.
- Do not greet unless greeted.
- Do not invent information.
- If answer not found, reply exactly:
Please contact the business directly.

Customer Question:
{user_message}

Answer:
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 80
                }
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()
        reply = result.get("response", "").strip()

        # Safety trim if model exceeds target
        words = reply.split()
        if len(words) > 40:
            reply = " ".join(words[:40])

        return jsonify({"reply": reply})

    except requests.exceptions.RequestException:
        return jsonify({"reply": "AI service is currently unavailable."}), 500

    except Exception:
        return jsonify({"reply": "An unexpected error occurred."}), 500


# ---------------------------------------------------
# Run
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
