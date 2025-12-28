"""
BƯỚC 3: ỨNG DỤNG GỢI Ý PHIM
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CLEANED_DATA_FILE = DATA_DIR / "movies_cleaned_for_bert.csv"
EMBEDDINGS_FILE = MODELS_DIR / "movie_embeddings.pkl"
EVAL_RESULTS_FILE = MODELS_DIR / "evaluation_results.pkl"

BERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

DEFAULT_NUM_RECOMMENDATIONS = 5
DEFAULT_MIN_YEAR = 1960
DEFAULT_MAX_YEAR = 2025
DEFAULT_MIN_RATING = 5.0

st.set_page_config(page_title="Movie AI Recommender", page_icon="🎬", layout="wide")


# ============================================================
# TẢI DỮ LIỆU
# ============================================================
@st.cache_resource
def load_assets():
    df = pd.read_csv(CLEANED_DATA_FILE)
    df["year"] = df["year"].fillna(0).astype(int)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    model = SentenceTransformer(BERT_MODEL_NAME)
    return df, embeddings, model


df, movie_embeddings, model = load_assets()
API_KEY = st.secrets["TMDB_API_KEY"]

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HÀM GỢI Ý PHIM (HYBRID SEARCH)
# ============================================================
def get_poster_url(movie_name, year):
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={API_KEY}&query={movie_name}&year={year}"
    try:
        response = requests.get(search_url, timeout=10)
        data = response.json()
        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"{POSTER_BASE_URL}{poster_path}"
    except:
        pass
    return PLACEHOLDER_POSTER


def recommend_movies(query, year_range, min_rating, num_results=5):
    # Keyword search
    keyword_mask = df["name"].str.contains(query, case=False, na=False)
    keyword_hits = df[keyword_mask].index.tolist()
    
    # Semantic search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]
    semantic_hits = similarities.argsort()[::-1].tolist()
    
    # Combine (keyword first)
    all_hits = list(dict.fromkeys(keyword_hits + semantic_hits))
    
    # Filter by context
    filtered = []
    for idx in all_hits:
        movie = df.iloc[idx]
        if year_range[0] <= movie["year"] <= year_range[1] and movie["rating"] >= min_rating:
            filtered.append(idx)
        if len(filtered) >= num_results:
            break
    
    return df.iloc[filtered] if filtered else pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🛠️ Điều khiển")
st.sidebar.subheader("📍 Bộ lọc")
year_range = st.sidebar.slider("Năm phát hành", DEFAULT_MIN_YEAR, DEFAULT_MAX_YEAR, (2000, DEFAULT_MAX_YEAR))
min_rating = st.sidebar.slider("Rating tối thiểu", 0.0, 10.0, DEFAULT_MIN_RATING)

st.sidebar.markdown("---")
st.sidebar.subheader("📜 Lịch sử")
if st.session_state.history:
    for item in reversed(st.session_state.history[-5:]):
        st.sidebar.write(f"• {item}")
    if st.sidebar.button("Xóa lịch sử"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.caption("Chưa có lịch sử.")


# ============================================================
# MAIN CONTENT
# ============================================================
st.title("🎬 Hệ thống Gợi ý Phim Thông minh")
tab1, tab2, tab3 = st.tabs(["🔍 Gợi ý phim", "📊 Phân tích dữ liệu", "🎯 Đánh giá"])

# --- TAB 1: GỢI Ý ---
with tab1:
    st.subheader("Tìm phim phù hợp với tâm trạng hoặc sở thích")
    user_input = st.text_input("Nhập tên phim hoặc mô tả:", placeholder="Ví dụ: Marvel, Phim thám hiểm...")
    
    if st.button("Tìm kiếm"):
        if user_input:
            if user_input not in st.session_state.history:
                st.session_state.history.append(user_input)
            
            with st.spinner("Đang tìm kiếm..."):
                results = recommend_movies(user_input, year_range, min_rating)
                
                if not results.empty:
                    st.success(f"Tìm thấy {len(results)} phim phù hợp!")
                    cols = st.columns(5)
                    for i, (_, movie) in enumerate(results.iterrows()):
                        with cols[i]:
                            st.image(get_poster_url(movie["name"], movie["year"]))
                            st.write(f"**{movie['name']}**")
                            st.caption(f"⭐ {movie['rating']} | 📅 {movie['year']}")
                            st.info(f"🎭 {movie['genre']}")
                            with st.expander("Tóm tắt"):
                                st.write(movie["description"])
                else:
                    st.warning("Không tìm thấy phim. Hãy thử nới lỏng bộ lọc!")

# --- TAB 2: VISUALIZATION ---
with tab2:
    st.header("📈 Trực quan hóa dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bố Rating")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["rating"], bins=20, kde=True, color="skyblue", ax=ax1)
        st.pyplot(fig1)
        
        st.subheader("Top 10 Phim điểm cao")
        top10 = df.nlargest(10, "rating").sort_values(by="rating")
        fig2, ax2 = plt.subplots()
        ax2.barh(top10["name"], top10["rating"], color="gold")
        plt.tight_layout()
        st.pyplot(fig2)
    
    with col2:
        st.subheader("Top 10 Thể loại")
        genres = df["genre"].str.split(", ").explode().value_counts().head(10).sort_values()
        fig3, ax3 = plt.subplots()
        genres.plot(kind="barh", color="salmon", ax=ax3)
        st.pyplot(fig3)
        
        st.subheader("Tương quan Rating - Year")
        fig4, ax4 = plt.subplots()
        sns.heatmap(df[["rating", "year"]].corr(), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

# --- TAB 3: EVALUATION ---
with tab3:
    st.header("🎯 Đánh giá Mô hình")
    
    # Load kết quả đánh giá
    try:
        with open(EVAL_RESULTS_FILE, "rb") as f:
            eval_results = pickle.load(f)
        
        st.markdown(f"Đánh giá trên **{eval_results['num_samples']} mẫu** từ tổng **{eval_results['total_movies']:,} phim**")
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision@5", f"{eval_results['precision_at_k']*100:.1f}%", help="Tỷ lệ phim gợi ý có cùng thể loại")
        m2.metric("Recall@5", f"{eval_results['recall_at_k']*100:.1f}%", help="Độ bao phủ thể loại")
        m3.metric("RMSE", f"{eval_results['rmse']:.3f}", help="Root Mean Square Error")
        m4.metric("MAE", f"{eval_results['mae']:.3f}", help="Mean Absolute Error")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Phân bố Sai số Rating")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.histplot(eval_results['errors'], bins=30, kde=True, color="steelblue", ax=ax1)
            ax1.axvline(x=0, color='red', linestyle='--', label='Zero Error')
            ax1.set_xlabel("Error (Actual - Predicted)")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            st.pyplot(fig1)
        
        with col2:
            st.subheader("📊 Phân bố Similarity Score")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(eval_results['similarities'], bins=30, kde=True, color="teal", ax=ax2)
            ax2.axvline(x=eval_results['similarities'].mean(), color='red', linestyle='--', label=f"Mean: {eval_results['similarities'].mean():.3f}")
            ax2.set_xlabel("Cosine Similarity")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            st.pyplot(fig2)
        
        # Giải thích
        st.markdown("---")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.success(
                "**✅ Ưu điểm:**\n"
                f"- Precision@5 cao ({eval_results['precision_at_k']*100:.1f}%): Gợi ý đúng thể loại\n"
                f"- Recall@5 tốt ({eval_results['recall_at_k']*100:.1f}%): Bao phủ nhiều thể loại\n"
                "- BERT hiểu ngữ nghĩa đa ngôn ngữ"
            )
        with col_info2:
            st.info(
                "**📝 Giải thích Metrics:**\n"
                "- **Precision@K**: % phim gợi ý có ít nhất 1 thể loại trùng\n"
                "- **Recall@K**: % thể loại gốc được cover bởi gợi ý\n"
                "- **RMSE/MAE**: Sai số dự đoán rating (thấp = tốt)"
            )
    
    except FileNotFoundError:
        st.warning("⚠️ Chưa có kết quả đánh giá. Chạy: `python step3_evaluation.py`")
