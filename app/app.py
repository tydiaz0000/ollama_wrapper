from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import ollama
from collections import OrderedDict
from ddgs import DDGS
import trafilatura
import os
import psycopg2
import json

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)
DB_CONFIG = {
    "host": "postgres",   # container name
    "database": "ai_requests",
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASS"),
    "port": 5432
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_table():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_request_log (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        session_id TEXT,
        prompt TEXT NOT NULL,

        http_status INTEGER,
        response_time_ms INTEGER,

        response_text TEXT,
        

        response_json JSONB,

        error_message TEXT
    );
                
    CREATE TABLE IF NOT EXISTS ai_chat_messages (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,              -- 'user' | 'assistant' | 'system'
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_session_id_created_at
    ON ai_chat_messages(session_id, created_at);


    """)

    conn.commit()
    cur.close()
    conn.close()

def save_log(
    prompt,
    session_id,
    http_status,
    response_time_ms,
    response_text,
    context,
    response_json,
    error_message
):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO ai_request_log (
        prompt,
        session_id,
        http_status,
        response_time_ms,
        response_text,
        context,
        response_json,
        error_message
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        prompt,
        session_id,
        http_status,
        response_time_ms,
        response_text,
        context,
        json.dumps(response_json) if response_json else None,
        error_message
    ))

    conn.commit()
    cur.close()
    conn.close()

client = ollama.Client(host="http://ollama:11434")

def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)

    return trafilatura.extract(downloaded)

def search_web(query, max_results=1):

    with DDGS() as ddgs:
        results = list(
            ddgs.text(query, max_results=max_results)
        )

    return results

def build_context(web_results):
    context_parts = []

    for result in web_results:
        text = extract_text_from_url(result["href"])

        context_parts.append(f"""
Title: {result['title']}

Snippet:
{text}

Source:
{result['href']}
""")
        
    return "\n\n".join(context_parts)

def get_session(session_id, limit=20):
    if not session_id:
        return []

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT role, content
        FROM ai_chat_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        LIMIT %s
    """, (session_id, limit))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]

def save_message(session_id, role, content):
    if not session_id:
        return

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO ai_chat_messages (session_id, role, content)
        VALUES (%s, %s, %s)
    """, (session_id, role, content))

    conn.commit()
    cur.close()
    conn.close()

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


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    model = data.get("model", "gemma3:1b-it-qat")
    prompt = data.get("prompt", "")
    system_prompt = data.get("system", "You are a helpful web search assistant.")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    print("Prompt: " + prompt)
    start_time = time.time()
    

    context = build_context(search_web(prompt))

    messages = []
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    messages.append({
        "role": "system",
        "content": "Use the following search results to answer the user prompt: " + context
    })
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    print("Context: " + context)
    response = client.chat(
        model=model,
        messages=messages
    )
    print("Response: " + response["message"]["content"])
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
    # RESPONSE
    # ----------------------------
    return jsonify({
        "model": model,
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
    })

    

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
    save_message(session_id, "system", system_prompt)
    # load session history if exists
    history = get_session(session_id)
    messages.extend(history)

    # add new user message
    messages.append({
        "role": "user",
        "content": prompt
    })
    save_message(session_id, "user", prompt)
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
    save_message(session_id, "assistant", assistant_message)

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
    # RESPONSE
    # ----------------------------

    response_json = jsonify({
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
    })

    save_log(prompt, session_id, 200, duration, assistant_message, None, response_json, None)

    return response_json


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
