import os
import json
import pickle
import threading
import faiss
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INDEX_FILE = "shl_vector_store.faiss"
METADATA_FILE = "shl_metadata.pkl"
JSON_DATA_FILE = "products.json" 
EMBED_MODEL = "all-MiniLM-L6-v2"

# 🔑 YOUR GEMINI API KEY (UNTOUCHED)
GEMINI_API_KEY = "AIzaSyD0DSk9ube3XUE2d3eBmSQl0lNuGeWtE-s"

genai.configure(api_key=GEMINI_API_KEY)

# --- AUTO-DISCOVER MODEL FUNCTION (UNTOUCHED) ---
def get_best_available_model():
    """
    Automatically finds a working model ID to prevent 404 errors.
    Prioritizes 'flash' > 'pro' > 'gemini-1.0'.
    """
    try:
        print("🔍 Auto-detecting available Gemini models...")
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
        
        if not available:
            print("⚠️ No models found! Check API Key permissions.")
            return "models/gemini-1.5-flash" # Fallback
            
        print(f"✅ Available Models: {available}")
        
        # Priority Logic: Prefer 1.5 Flash for speed and context window
        for m in available:
            if "gemini-1.5-flash" in m: return m
        for m in available:
            if "gemini-1.5-pro" in m: return m
        
        return available[0]
        
    except Exception as e:
        print(f"⚠️ Model detection failed: {e}. Defaulting to fallback.")
        return "models/gemini-1.5-flash"

# Initialize with the auto-discovered model
SELECTED_MODEL = get_best_available_model()
print(f"🚀 Using Gemini Model: {SELECTED_MODEL}")

# --- MODEL CONFIGURATION (UNTOUCHED) ---
gemini_model = genai.GenerativeModel(
    model_name=SELECTED_MODEL,
    generation_config={
        "temperature": 0.5,       
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048, 
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# ---------------- FASTAPI ----------------
app = FastAPI(title="SHL Recommendation Engine")

index = None
products = []
embedder = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 6
    detail_level: str = "Standard" 

@app.on_event("startup")
def load_all():
    global index, products, embedder

    print("⏳ Loading AI Resources...")
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # 1. Load FAISS Index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        print("⚠️ Index not found. Creating empty index.")
        index = faiss.IndexFlatL2(384)

    # 2. Load Metadata (Try Pickle -> Fallback to JSON)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            products = pickle.load(f)
    elif os.path.exists(JSON_DATA_FILE):
        print(f"⚠️ Pickle not found. Loading from {JSON_DATA_FILE}...")
        with open(JSON_DATA_FILE, "r", encoding="utf-8") as f:
            products = json.load(f)
    else:
        print("❌ No data file found (products.json or .pkl).")
        products = []

    print(f"✅ Backend ready with {len(products)} items.")

# ---------------- GEMINI CALL (UNTOUCHED) ----------------
def call_gemini(prompt, timeout=60): 
    result = {"text": None}

    def task():
        try:
            r = gemini_model.generate_content(prompt, stream=False)
            if r.prompt_feedback and r.prompt_feedback.block_reason:
                result["text"] = f"⚠️ Response blocked by AI safety filters. Reason: {r.prompt_feedback.block_reason}"
            else:
                result["text"] = r.text
        except Exception as e:
            print(f"Gemini Error Details: {e}")
            result["text"] = f"⚠️ Gemini processing error: {str(e)}"

    t = threading.Thread(target=task)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return "⚠️ Analysis timed out."
    return result["text"]

def generate_strategy_text(query, items, detail_level="Standard"):
    # ... (Prompt logic remains same) ...
    context = ""
    for i, it in enumerate(items):
        t_type = it.get('test_type', [])
        if isinstance(t_type, list): t_type = ", ".join(t_type)
        context += f"{i+1}. {it['name']} (Type: {t_type})\n   Desc: {it['description']}\n   URL: {it['url']}\n\n"

    detail_instruction = "Provide a standard professional summary."
    if detail_level == "Deep Dive":
        detail_instruction = "Provide an extensive, deep analysis with specific interview questions."
    elif detail_level == "Executive Summary":
        detail_instruction = "Provide a concise, high-level summary only."

    prompt = f"""
You are an expert SHL Consultant. Create a strategic assessment plan for: "{query}".
CONTEXT: The user needs a hiring strategy based ONLY on the SHL assessments listed below.
{context}
INSTRUCTIONS:
1. Write a strategy with 3 distinct sections.
2. BE CONCISE.
3. For Section 2 (Recommendations), select exactly 3-4 top tests.
4. Explain the 'Why' for each test.
"""
    return call_gemini(prompt, timeout=60)

# --- HELPER: SEARCH FUNCTION ---
def perform_search(query, k=10):
    if index and index.ntotal > 0:
        qv = embedder.encode([query]).astype("float32")
        _, ids = index.search(qv, k)
        
        seen = set()
        results = []
        for idx in ids[0]:
            if idx == -1 or idx >= len(products): continue
            it = products[idx]
            if it["url"] in seen: continue
            seen.add(it["url"])
            results.append(it)
        return results
    return []

# ==========================================
# 1. MANDATORY HEALTH CHECK (PDF Appendix 2)
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ==========================================
# 2. STRICT SUBMISSION ENDPOINT (PDF Appendix 2)
# ==========================================
@app.post("/recommend")
def recommend_strict(req: QueryRequest):
    """
    Strict JSON output for the assignment evaluation bot.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Search
    raw_results = perform_search(req.query, k=req.top_k)
    
    # Format strictly as per PDF Appendix 2
    formatted_output = []
    for item in raw_results[:10]: # Max 10 per PDF
        formatted_output.append({
            "url": item.get("url", ""),
            "name": item.get("name", ""),
            "adaptive_support": item.get("adaptive_support", "No"),
            "description": item.get("description", "")[:300],
            "duration": int(float(item.get("duration", 0) or 0)),
            "remote_support": item.get("remote_support", "Yes"),
            "test_type": item.get("test_type", []) if isinstance(item.get("test_type"), list) else [item.get("test_type")]
        })

    return {"recommended_assessments": formatted_output}

# ==========================================
# 3. RICH UI ENDPOINT (For Streamlit)
# ==========================================
@app.post("/strategy")
def recommend_strategy(req: QueryRequest):
    """
    Rich output with Gemini analysis for the Streamlit UI.
    """
    raw_results = perform_search(req.query, k=req.top_k)
    ai_text = generate_strategy_text(req.query, raw_results, req.detail_level)
    
    return {
        "ai_response": ai_text,
        "raw_results": raw_results
    }