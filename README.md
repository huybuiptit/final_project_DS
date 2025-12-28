# 🎬 Movie Recommendation System

Hệ thống gợi ý phim thông minh sử dụng **BERT Embeddings** và **Hybrid Search**.

## 📁 Cấu trúc Project

```
final_project_DS/
├── data/                         # Dữ liệu
│   ├── imdb_tmdb_5000_clean.csv  # Dữ liệu gốc
│   └── movies_cleaned_for_bert.csv
├── models/                       # Embeddings
│   └── movie_embeddings.pkl
├── step1_data_cleaning.py        # Bước 1: Làm sạch dữ liệu
├── step2_embedding.py            # Bước 2: Tạo BERT embeddings
├── app.py                        # Bước 3: Web app (Streamlit)
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
```

## 📦 Các bước chạy

### Bước 1: Làm sạch dữ liệu

```bash
python step1_data_cleaning.py
```

### Bước 2: Tạo embeddings

```bash
python step2_embedding.py
```

### Bước 3: Đánh giá mô hình

```bash
python step3_evaluation.py
```

### Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

## ⚙️ Cấu hình API Key

Tạo file `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_api_key_here"
```

## 📊 Tính năng

- **Hybrid Search**: Kết hợp keyword + semantic search
- **BERT Embeddings**: Hiểu ngữ nghĩa mô tả phim
- **Context-Aware**: Lọc theo năm, rating
- **Visualization**: Biểu đồ phân tích dữ liệu
- **Model Evaluation**: Đánh giá hiệu năng
