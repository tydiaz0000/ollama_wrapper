from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import ollama
from collections import OrderedDict
from duckduckgo_search import DDGS
import trafilatura

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)


# ----------------------------
# SESSION STORAGE (max 5)
# ----------------------------
SESSIONS = OrderedDict()
MAX_SESSIONS = 5

client = ollama.Client(host="http://ollama:11434")
def get_session(session_id):
    if not session_id:
        return None
    return SESSIONS.get(session_id)


def update_session(session_id, messages):
    if not session_id:
        return

    # If session exists, update order (LRU behavior)
    if session_id in SESSIONS:
        SESSIONS.move_to_end(session_id)

    SESSIONS[session_id] = messages

    # Enforce max sessions (LRU eviction)
    if len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


# ----------------------------
# MAIN ENDPOINT
# ----------------------------
@app.route("/models", methods=["GET"])

def list_models():
    try:
        client = ollama.Client(host="http://ollama:11434")
        response = client.list()

        models = []

        for m in response.get("models", []):

            details = m.details

            models.append({
                "name": getattr(m, "model", None),
                "digest": getattr(m, "digest", None),
                "size": getattr(m, "size", None),
                "modified_at": str(getattr(m, "modified_at", None)),
                "details": {
                    "format": getattr(details, "format", None),
                    "family": getattr(details, "family", None),
                    "families": getattr(details, "families", None),
                    "parameter_size": getattr(details, "parameter_size", None),
                    "quantization_level": getattr(details, "quantization_level", None),
                }
            })

        return jsonify({
            "count": len(models),
            "models": models
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])


def chat():
    data = request.get_json()

    model = data.get("model", "gemma3:1b-it-qat")
    prompt = data.get("prompt", "")
    system_prompt = data.get("system", "You are a helpful assistant.")
    session_id = data.get("session_id")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    start_time = time.time()

    # ----------------------------
    # BUILD MESSAGE HISTORY
    # ----------------------------
    messages = []

    # system prompt
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # load session history if exists
    history = get_session(session_id)
    if history:
        messages.extend(history)

    # add new user message
    messages.append({
        "role": "user",
        "content": prompt
    })

    # ----------------------------
    # CALL OLLAMA CHAT
    # ----------------------------
    response = client.chat(
        model=model,
        messages=messages
    )

    end_time = time.time()
    duration = end_time - start_time

    assistant_message = response["message"]["content"]

    # ----------------------------
    # TOKEN METRICS (if available)
    # ----------------------------
    usage = response.get("usage", {}) or {}

    prompt_eval_count = response.get("prompt_eval_count")
    eval_count = response.get("eval_count")
    prompt_eval_duration = response.get("prompt_eval_duration")
    eval_duration = response.get("eval_duration")
    load_duration = response.get("load_duration")

    tokens_per_sec = None
    if eval_count and duration > 0:
        tokens_per_sec = eval_count / duration

    # ----------------------------
    # UPDATE SESSION
    # ----------------------------
    if session_id:
        new_history = messages[1:] + [{
            "role": "assistant",
            "content": assistant_message
        }]
        update_session(session_id, new_history)

    # ----------------------------
    # RESPONSE
    # ----------------------------
    return jsonify({
        "model": model,
        "session_id": session_id,
        "response": assistant_message,
        "timing": {
            "total_seconds": duration,
            "tokens_per_second": tokens_per_sec
        },
        "tokens": {
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_duration": eval_duration,
            "load_duration": load_duration
        },
        "session_active": bool(session_id),
        "session_count": len(SESSIONS)
    })


# ----------------------------
# OPTIONAL: VIEW SESSIONS
# ----------------------------
@app.route("/sessions", methods=["GET"])
def list_sessions():
    return jsonify({
        "active_sessions": list(SESSIONS.keys()),
        "count": len(SESSIONS)
    })


# ----------------------------
# OPTIONAL: CLEAR SESSION
# ----------------------------
@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return jsonify({"deleted": session_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# --------------------------------------------------
# AFTER REQUEST
# --------------------------------------------------
@app.after_request
def add_cors_headers(response):

    response.headers["Access-Control-Allow-Origin"] = "*"

    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization"
    )

    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    return response
