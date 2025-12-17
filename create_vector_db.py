import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURATION ---
DATA_FILE = "products.json"
INDEX_FILE = "shl_vector_store.faiss"
METADATA_FILE = "shl_metadata.pkl" 
MODEL_NAME = "all-MiniLM-L6-v2" # Fast, free, standard for RAG

def build_vector_db():
    print("🚀 Starting Vector Database Build...")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found. Please run the scraper first.")
        return
        
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    print(f"📉 Loaded {len(products)} items from {DATA_FILE}")

    # 2. Prepare Text for Embedding
    # This is the "Context Engineering" part.
    # We combine the most important fields so the AI sees the full picture.
    # Format: "Test Type: [K]. Name: [Java]. Description: [Measures coding...]"
    corpus = []
    for item in products:
        # Convert list to string for test_type (e.g., ['K', 'A'] -> "K, A")
        t_type_str = ", ".join(item.get('test_type', []))
        text = f"Test Type: {t_type_str}. Name: {item['name']}. Description: {item['description']}"
        corpus.append(text)
    
    # 3. Initialize Model
    print(f"🧠 Loading AI Model ({MODEL_NAME})...")
    # This will download ~80MB the first time you run it
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Generate Embeddings
    print("⚡ Generating Embeddings (This takes about 30-60 seconds)...")
    embeddings = model.encode(corpus, show_progress_bar=True)
    
    # Convert to numpy float32 (Required for FAISS)
    embeddings = np.array(embeddings).astype("float32")
    
    # 5. Build FAISS Index
    dimension = embeddings.shape[1] # 384 dimensions for MiniLM
    print(f"🏗️  Building FAISS Index (Dimension: {dimension})...")
    
    # IndexFlatL2 = Exact Search (Perfect for <10k items)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"✅ Added {index.ntotal} vectors to index.")

    # 6. Save Artifacts
    faiss.write_index(index, INDEX_FILE)
    
    # We save the raw product list as a pickle file.
    # Why? FAISS only returns an ID (e.g., "Item 42"). 
    # We need this file to look up what "Item 42" actually is (Name, URL, Duration).
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(products, f)
        
    print("\n🎉 Success! AI Engine Built.")
    print(f"   📂 Index: {INDEX_FILE}")
    print(f"   📂 Metadata: {METADATA_FILE}")
    print("   👉 Ready for Backend API.")

if __name__ == "__main__":
    build_vector_db()