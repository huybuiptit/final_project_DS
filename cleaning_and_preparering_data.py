import pandas as pd
df = pd.read_csv('D:/final_project_DS/imdb_tmdb_5000_clean.csv')

# 2. Loại bỏ các bộ phim trùng lặp dựa trên tiêu đề (name)
# Điều này đảm bảo mỗi bộ phim chỉ xuất hiện một lần trong hệ thống
df = df.drop_duplicates(subset=['name'], keep='first')

# 3. Xử lý các giá trị thiếu (Handling Missing Values)
# BERT không thể xử lý giá trị Null, nên ta cần chuyển chúng về chuỗi rỗng hoặc giá trị mặc định
df['genre'] = df['genre'].fillna('')
df['actors'] = df['actors'].fillna('')
df['director'] = df['director'].fillna('Unknown')

# Riêng cột Year, ta chuyển về kiểu số nguyên (int) để hiển thị đẹp hơn
df['year'] = df['year'].fillna(0).astype(int)

# 4. Tạo cột "Feature Tổng hợp" (Combined Features)
# Đây là bước "Advanced Embedding" giúp BERT học được mối quan hệ giữa các thông tin khác nhau
def create_combined_features(row):
    # Nối các trường thông tin thành một đoạn văn bản hoàn chỉnh
    return (f"Title: {row['name']}. "
            f"Genre: {row['genre']}. "
            f"Director: {row['director']}. "
            f"Actors: {row['actors']}. "
            f"Plot: {row['description']}")

# Áp dụng hàm trên cho từng dòng trong DataFrame
df['combined_features'] = df.apply(create_combined_features, axis=1)

# 5. Kiểm tra kết quả sau khi làm sạch
print(f"Số lượng phim sau khi làm sạch: {len(df)}")
print("\nVí dụ về dữ liệu tổng hợp cho BERT:")
print(df['combined_features'].iloc[0])

# 6. Lưu file đã làm sạch để dùng cho bước tạo Embedding
df.to_csv('movies_cleaned_for_bert.csv', index=False, encoding='utf-8-sig')
print("\n✅ Đã lưu file: movies_cleaned_for_bert.csv")