
import requests
import pandas as pd
import time
import os

# CẤU HÌNH

API_KEY = "YOUR_TMDB_API_KEY_HERE"  # Thay key của bạn vào đây
BASE_URL = "https://api.themoviedb.org/3"
TARGET_MOVIES = 5000
MOVIES_PER_PAGE = 20
TOTAL_PAGES = TARGET_MOVIES // MOVIES_PER_PAGE  # 5000 / 20 = 250 trang

# Thư mục lưu dữ liệu
DATA_DIR = "data"
OUTPUT_FILE = "imdb_tmdb_5000_clean.csv"


# CÁC HÀM HỖ TRỢ

def get_genre_mapping():

    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()
    return {g['id']: g['name'] for g in response.get('genres', [])}


def get_movie_credits(movie_id):
    
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
    try:
        res = requests.get(url).json()
        
        # Lấy Đạo diễn
        director = next(
            (member['name'] for member in res.get('crew', []) 
             if member['job'] == 'Director'), 
            "Unknown"
        )
        
        # Lấy Top 3 Diễn viên
        actors = ", ".join([member['name'] for member in res.get('cast', [])[:3]])
        
        return director, actors
    except:
        return "Unknown", "Unknown"


# PIPELINE CRAWL DỮ LIỆU

def crawl_movies():
    
    genres_dict = get_genre_mapping()
    all_data = []
    
    print("=" * 60)
    print(f"🚀 BẮT ĐẦU THU THẬP {TARGET_MOVIES} PHIM TỪ TMDB")
    print("=" * 60)
    
    for page in range(1, TOTAL_PAGES + 1):
        # 1. Lấy danh sách phim phổ biến theo từng trang
        list_url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        
        try:
            list_res = requests.get(list_url).json()
            movies = list_res.get('results', [])
            
            for m in movies:
                m_id = m['id']
                
                # 2. Gọi API phụ để lấy Director và Actors
                director, actors = get_movie_credits(m_id)
                
                # 3. Tổng hợp 7 features
                movie_info = {
                    'name': m.get('title'),
                    'description': m.get('overview'),
                    'genre': ", ".join([genres_dict.get(id, "") for id in m.get('genre_ids', [])]),
                    'rating': m.get('vote_average'),
                    'director': director,
                    'actors': actors,
                    'year': m.get('release_date', '')[:4]
                }
                
                # Chỉ lưu nếu có đủ mô tả để chạy BERT sau này
                if movie_info['description'] and len(movie_info['description']) > 30:
                    all_data.append(movie_info)
            
            # In tiến độ
            if page % 10 == 0:
                print(f"✅ Đã xử lý xong trang {page}/{TOTAL_PAGES} (Thu được {len(all_data)} phim)")
            
            # Nghỉ ngắn để không bị TMDB chặn (Rate limit)
            time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Lỗi ở trang {page}: {e}")
            continue
    
    # Tạo DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\n✅ Hoàn thành crawl! Thu được {len(df)} phim.")
    
    return df


def save_data(df, filename=OUTPUT_FILE):
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"💾 Đã lưu dữ liệu vào: {filepath}")


# MAIN

if __name__ == "__main__":
    print("=" * 60)
    print("CRAWL DỮ LIỆU PHIM TỪ TMDB API")
    print("=" * 60)
    
    # Kiểm tra API key
    if API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("\n⚠️  CẢNH BÁO: Bạn chưa cấu hình API key!")

        print("\n" + "=" * 60)
        
        
        df_sample = pd.DataFrame(sample_data)
        print("\nCấu trúc dữ liệu (7 features):")
        print(df_sample.to_string())
        
    else:
        # Crawl dữ liệu thực
        df = crawl_movies()
        
        # Hiển thị thông tin
        print("\n📊 THỐNG KÊ DỮ LIỆU:")
        print(f"   - Số lượng phim: {len(df)}")
        print(f"   - Các cột: {list(df.columns)}")
        print(f"\n   - 5 phim đầu tiên:")
        print(df.head().to_string())
        
        # Lưu dữ liệu
        save_data(df)
        
        print("\n" + "=" * 60)
        print("🎉 HOÀN THÀNH CRAWL DỮ LIỆU!")
        print("=" * 60)
    
