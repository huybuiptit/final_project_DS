"""
ĐÁNH GIÁ MÔ HÌNH GỢI Ý PHIM
Tính: Precision@K, Recall@K, RMSE, MAE
Chạy: python step3_evaluation.py
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CLEANED_DATA_FILE = DATA_DIR / "movies_cleaned_for_bert.csv"
EMBEDDINGS_FILE = MODELS_DIR / "movie_embeddings.pkl"
EVAL_RESULTS_FILE = MODELS_DIR / "evaluation_results.pkl"

K = 5  # Top-K recommendations
NUM_TEST_SAMPLES = 500  # Số phim để test


# ============================================================
# TẢI DỮ LIỆU
# ============================================================
def load_data():
    df = pd.read_csv(CLEANED_DATA_FILE)
    df["year"] = df["year"].fillna(0).astype(int)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return df, embeddings


# ============================================================
# HÀM ĐÁNH GIÁ
# ============================================================
def get_genres_set(genre_str):
    """Chuyển chuỗi thể loại thành set."""
    if pd.isna(genre_str) or genre_str == "":
        return set()
    return set(g.strip() for g in genre_str.split(","))


def calculate_precision_recall_at_k(df, embeddings, k=5, num_samples=500):
    """Tính Precision@K và Recall@K dựa trên thể loại."""
    np.random.seed(42)
    test_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    
    precisions = []
    recalls = []
    
    for idx in test_indices:
        # Lấy thể loại của phim gốc
        original_genres = get_genres_set(df.iloc[idx]["genre"])
        if not original_genres:
            continue
        
        # Tìm K phim tương tự nhất (không tính chính nó)
        query_embedding = embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        similar_indices = similarities.argsort()[::-1][1:k+1]  # Bỏ chính nó
        
        # Tính precision và recall
        matched_genres = set()
        relevant_count = 0
        
        for sim_idx in similar_indices:
            rec_genres = get_genres_set(df.iloc[sim_idx]["genre"])
            if rec_genres & original_genres:  # Có ít nhất 1 thể loại trùng
                relevant_count += 1
            matched_genres.update(rec_genres & original_genres)
        
        precision = relevant_count / k
        recall = len(matched_genres) / len(original_genres) if original_genres else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.mean(precisions), np.mean(recalls)


def calculate_rating_metrics(df, embeddings, k=5, num_samples=500):
    """Tính RMSE và MAE dựa trên rating của phim tương tự."""
    np.random.seed(42)
    test_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    
    errors = []
    
    for idx in test_indices:
        original_rating = df.iloc[idx]["rating"]
        
        # Tìm K phim tương tự nhất
        query_embedding = embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        similar_indices = similarities.argsort()[::-1][1:k+1]
        
        # Dự đoán rating = trung bình rating của K phim tương tự (có trọng số)
        sim_ratings = df.iloc[similar_indices]["rating"].values
        sim_weights = similarities[similar_indices]
        
        if sim_weights.sum() > 0:
            predicted_rating = np.average(sim_ratings, weights=sim_weights)
        else:
            predicted_rating = sim_ratings.mean()
        
        errors.append(original_rating - predicted_rating)
    
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    
    return rmse, mae, errors


def calculate_similarity_distribution(embeddings, num_samples=1000):
    """Tính phân bố similarity scores."""
    np.random.seed(42)
    indices = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
    
    all_similarities = []
    for idx in indices:
        query_embedding = embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_sims = np.sort(similarities)[::-1][1:6]  # Top 5 (không tính chính nó)
        all_similarities.extend(top_sims)
    
    return np.array(all_similarities)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 50)
    print("ĐÁNH GIÁ MÔ HÌNH GỢI Ý PHIM")
    print("=" * 50)
    
    # Tải dữ liệu
    print("\n[1] Đang tải dữ liệu...")
    df, embeddings = load_data()
    print(f"    Số phim: {len(df)}")
    print(f"    Embedding shape: {embeddings.shape}")
    
    # Tính Precision@K và Recall@K
    print(f"\n[2] Tính Precision@{K} và Recall@{K}...")
    precision, recall = calculate_precision_recall_at_k(df, embeddings, K, NUM_TEST_SAMPLES)
    print(f"    Precision@{K}: {precision:.4f} ({precision*100:.2f}%)")
    print(f"    Recall@{K}: {recall:.4f} ({recall*100:.2f}%)")
    
    # Tính RMSE và MAE
    print(f"\n[3] Tính RMSE và MAE...")
    rmse, mae, errors = calculate_rating_metrics(df, embeddings, K, NUM_TEST_SAMPLES)
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    # Tính phân bố similarity
    print(f"\n[4] Tính phân bố similarity...")
    similarities = calculate_similarity_distribution(embeddings)
    print(f"    Mean similarity: {similarities.mean():.4f}")
    print(f"    Std similarity: {similarities.std():.4f}")
    
    # Lưu kết quả
    results = {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "rmse": rmse,
        "mae": mae,
        "k": K,
        "num_samples": NUM_TEST_SAMPLES,
        "errors": errors,
        "similarities": similarities,
        "total_movies": len(df)
    }
    
    with open(EVAL_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Đã lưu kết quả: {EVAL_RESULTS_FILE}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    main()
