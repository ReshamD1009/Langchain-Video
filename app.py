import os
from dotenv import load_dotenv
from loaders.youtube_loader import load_youtube_texts
from langchain_community.vectorstores import PGVector
from embeddings.youtube_embedding import store_embeddings_pgvector, similarity_search_with_response
from yt_dlp import YoutubeDL
from flask import Flask, render_template, request, jsonify, g

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def validate_env_variables():
    if not DB_CONNECTION_URL_2:
        raise ValueError("Database connection URL is missing!")
    print("Database connection URL loaded successfully.")
    
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is missing!")
    print("Google API key loaded successfully.")
    
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing!")
    print("Groq API key loaded successfully.")

validate_env_variables()

# YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=1bUy-1hGZpI"
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=JLDLANs_m_w"

app = Flask(__name__)

# Load and embed texts on startup
def initialize_chatbot():
    try:
        texts = load_youtube_texts(YOUTUBE_VIDEO_URL)
        if not texts:
            print("No chunks generated from the video text.")
            return False
        store_embeddings_pgvector(texts)
        return True
    except Exception as e:
        print(f"Error in initialization: {e}")
        return False

def ensure_embeddings():
    if not hasattr(g, 'embeddings_loaded'):
        g.embeddings_loaded = initialize_chatbot()  #Uses Flask's g object to track initialization state

@app.route('/')
def index():
    ensure_embeddings()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])  #for handling user queries
def chat():
    ensure_embeddings()
    try:
        user_query = request.form.get('query', '')
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        response = similarity_search_with_response(user_query)
        
        # Convert response to string if it's not already a string
        if hasattr(response, 'content'):
            response = response.content
        
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
   
    app.run(debug=True)