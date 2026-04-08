from flask import Flask, request
from pipeline.request_handler import handle_ask

app = Flask(__name__)

# ---- Endpoint ----
@app.route("/ask", methods=["POST"])
def ask():
    return handle_ask(request)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)

