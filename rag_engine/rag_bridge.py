from flask import Flask, request, jsonify
from rag_query import ask_rag 
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_DIR = "./conversation_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

app = Flask(__name__)

# Configure request timeout to 55 seconds (less than Rasa's 60s timeout)
app.config['PROPAGATE_EXCEPTIONS'] = True

@app.route('/ask', methods=['POST'])
def handle_query():
    try:
        data = request.json
        question = data.get("question")
        session_id = data.get("session_id", "anonymous")
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        logger.info(f"RAG query received: {question[:80]}...")
        
        # Call ask_rag with built-in error handling
        result = ask_rag(question, session_id) 
        
        logger.info(f"RAG response: {result.get('answer', '')[:60]}...")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Bridge error: {str(e)}")
        return jsonify({
            "answer": "La base de connaissances est temporairement indisponible.",
            "source": None,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    logger.info("Starting RAG bridge on port 5006...")
    app.run(host='0.0.0.0', port=5006, threaded=True)