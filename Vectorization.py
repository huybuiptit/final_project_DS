from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import time

# 1. Load dữ liệu đã làm sạch
df = pd.read_csv('movies_cleaned_for_bert.csv')

# 2. Khởi tạo mô hình BERT (Pre-trained)
# 'all-MiniLM-L6-v2' là mô hình cực kỳ phổ biến: nhanh, nhẹ nhưng hiểu ngữ nghĩa rất sâu
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"🚀 Đang bắt đầu tạo Embeddings cho {len(df)} phim...")
start_time = time.time()

# 3. THỰC HIỆN VECTOR HÓA (Đây là bước "Advanced Embedding")
# Chuyển toàn bộ cột 'combined_features' thành các Vector 384 chiều
movie_embeddings = model.encode(df['combined_features'].tolist(), show_progress_bar=True)

# 4. Lưu kết quả
# Lưu ma trận Embedding vào file .pkl để không phải chạy lại (vì bước này tốn CPU/GPU)
with open('movie_embeddings.pkl', 'wb') as f:
    pickle.dump(movie_embeddings, f)

end_time = time.time()
print(f"✅ Hoàn thành trong {(end_time - start_time)/60:.2f} phút!")
print(f"Kích thước ma trận: {movie_embeddings.shape}") 
# Kết quả thường là (4455, 384) -> 4455 phim, mỗi phim là 1 vector 384 số.