"""
BƯỚC 1: LÀM SẠCH DỮ LIỆU
Chạy: python step1_data_cleaning.py
"""
import pandas as pd
from pathlib import Path

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "imdb_tmdb_5000_clean.csv"
CLEANED_DATA_FILE = DATA_DIR / "movies_cleaned_for_bert.csv"


# ============================================================
# HÀM XỬ LÝ
# ============================================================
def load_raw_data():
    """Đọc dữ liệu thô."""
    return pd.read_csv(RAW_DATA_FILE)


def remove_duplicates(df):
    """Loại bỏ phim trùng lặp."""
    return df.drop_duplicates(subset=["name"], keep="first")


def handle_missing_values(df):
    """Xử lý giá trị thiếu."""
    df["genre"] = df["genre"].fillna("")
    df["actors"] = df["actors"].fillna("")
    df["director"] = df["director"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    df["year"] = df["year"].fillna(0).astype(int)
    return df


def create_combined_features(row):
    """Tạo chuỗi tổng hợp cho BERT."""
    return (
        f"Title: {row['name']}. "
        f"Genre: {row['genre']}. "
        f"Director: {row['director']}. "
        f"Actors: {row['actors']}. "
        f"Plot: {row['description']}"
    )


def add_combined_features(df):
    """Thêm cột combined_features."""
    df["combined_features"] = df.apply(create_combined_features, axis=1)
    return df


def save_cleaned_data(df):
    """Lưu dữ liệu đã làm sạch."""
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CLEANED_DATA_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 50)
    print("BƯỚC 1: LÀM SẠCH DỮ LIỆU")
    print("=" * 50)
    
    # Pipeline xử lý
    df = load_raw_data()
    print(f"[1] Đọc dữ liệu: {len(df)} bản ghi")
    
    df = remove_duplicates(df)
    print(f"[2] Loại bỏ trùng lặp: {len(df)} bản ghi")
    
    df = handle_missing_values(df)
    print(f"[3] Xử lý giá trị thiếu: OK")
    
    df = add_combined_features(df)
    print(f"[4] Tạo combined_features: OK")
    
    save_cleaned_data(df)
    print(f"\n✅ Hoàn thành! Đã lưu: {CLEANED_DATA_FILE}")
    print(f"   Số lượng phim: {len(df)}")


if __name__ == "__main__":
    main()
