import pandas as pd
import json
import glob

# 1. Lấy 5 file batch để đảm bảo có ít nhất 2000-5000 phim
file_paths = sorted(glob.glob(r"D:/final_project_DS/movies_batch_*.json"))[:3]

parsed_movies = []

for path in file_paths:
    with open(path, 'r', encoding='utf-8') as f:
        try:
            batch = json.load(f)
            for movie in batch:
                # Hàm hỗ trợ trích xuất dữ liệu từ List hoặc Dict
                def get_val(data, field):
                    val = data.get(field, "")
                    if isinstance(val, list) and len(val) > 0:
                        if isinstance(val[0], dict): return val[0].get('name', '')
                        return ", ".join([str(i) for i in val])
                    return val if val else ""

                # Trích xuất 7 features
                res = {
                    'name': movie.get('name', 'N/A'),
                    'description': movie.get('description', ''),
                    'genre': get_val(movie, 'genre'),
                    'rating': movie.get('aggregateRating', {}).get('ratingValue', 0) if isinstance(movie.get('aggregateRating'), dict) else 0,
                    'director': get_val(movie, 'director'),
                    'actors': get_val(movie, 'actor'),
                    'year': str(movie.get('datePublished', ''))[:4]
                }
                
                # Chỉ thêm vào nếu có Description (để chạy BERT) và Name
                if res['description'] and len(res['description']) > 10:
                    parsed_movies.append(res)
        except Exception as e:
            print(f"Lỗi tại file {path}: {e}")

# 2. Tạo DataFrame
df = pd.DataFrame(parsed_movies)

# 3. Hậu xử lý: Ép kiểu và lọc số lượng
if not df.empty:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df = df.drop_duplicates(subset=['name']) # Xóa phim trùng
    print(f"✅ Thành công! Tổng số phim thu được: {len(df)}")
    print(df[['name', 'director', 'year']].head())
    
    # Lưu file để làm đồ án
    df.to_csv("imdb_7_features.csv", index=False)
else:
    print("❌ Vẫn chưa lấy được dữ liệu. Hãy kiểm tra lại đường dẫn file JSON.")