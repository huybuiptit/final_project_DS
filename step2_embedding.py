"""
BƯỚC 2: TẠO BERT EMBEDDINGS
Chạy: python step2_embedding.py
"""
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CLEANED_DATA_FILE = DATA_DIR / "movies_cleaned_for_bert.csv"
EMBEDDINGS_FILE = MODELS_DIR / "movie_embeddings.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# HÀM XỬ LÝ
# ============================================================
def load_cleaned_data():
    """Đọc dữ liệu đã làm sạch."""
    df = pd.read_csv(CLEANED_DATA_FILE)
    df["year"] = df["year"].fillna(0).astype(int)
    return df


def load_model():
    """Tải model BERT."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(model, texts):
    """Tạo embeddings từ văn bản."""
    return model.encode(texts, show_progress_bar=True)


def save_embeddings(embeddings):
    """Lưu embeddings vào file."""
    MODELS_DIR.mkdir(exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 50)
    print("BƯỚC 2: TẠO BERT EMBEDDINGS")
    print("=" * 50)
    
    # Đọc dữ liệu
    df = load_cleaned_data()
    print(f"[1] Đọc dữ liệu: {len(df)} phim")
    
    # Tải model
    print(f"[2] Tải model: {MODEL_NAME}")
    model = load_model()
    
    # Tạo embeddings
    print(f"[3] Tạo embeddings...")
    texts = df["combined_features"].tolist()
    embeddings = generate_embeddings(model, texts)
    
    # Lưu
    save_embeddings(embeddings)
    print(f"\n✅ Hoàn thành! Đã lưu: {EMBEDDINGS_FILE}")
    print(f"   Kích thước: {embeddings.shape}")


if __name__ == "__main__":
    main()
