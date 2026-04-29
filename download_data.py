import os
import sys
import subprocess
import zipfile

def download_data():
    """
    Tải dataset cho SemEval-2026 Task 13.
    Mặc định thử tải từ Kaggle CLI. 
    Nếu không có Kaggle, bạn có thể cấu hình URL tải trực tiếp từ GitHub/Drive.
    """
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Cấu hình Kaggle (Nếu dùng Kaggle API)
    # Tên competition: sem-eval-2026-task-13-subtask-a
    COMP_NAME = "sem-eval-2026-task-13-subtask-a"

    print(f"--- Đang kiểm tra dữ liệu trong thư mục '{DATA_DIR}' ---")
    
    # Kiểm tra xem file đã tồn tại chưa
    if os.path.exists(os.path.join(DATA_DIR, "train.parquet")):
        print("✓ Dữ liệu đã tồn tại. Bỏ qua bước tải.")
        return

    # Thử dùng Kaggle CLI
    try:
        print(f"Đang thử tải từ Kaggle competition: {COMP_NAME}...")
        # Lệnh: kaggle competitions download -c [COMP_NAME] -p [DATA_DIR]
        subprocess.run(["kaggle", "competitions", "download", "-c", COMP_NAME, "-p", DATA_DIR], check=True)
        
        zip_path = os.path.join(DATA_DIR, f"{COMP_NAME}.zip")
        if os.path.exists(zip_path):
            print(f"Đang giải nén {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)
            print("✓ Giải nén thành công.")
        else:
            # Có thể Kaggle tải lẻ từng file parquet
            print("✓ Tải hoàn tất (không thấy file zip, có thể đã là file lẻ).")
            
    except Exception as e:
        print(f"⚠️ Không thể tải từ Kaggle CLI: {e}")
        print("\n--- Phương án thay thế: Tải thủ công ---")
        print("1. Truy cập: https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-a/data")
        print("2. Tải các file: train.parquet, validation.parquet, test.parquet")
        print(f"3. Đặt chúng vào thư mục: {os.path.abspath(DATA_DIR)}")
        
        # Nếu bạn có link GitHub cụ thể, có thể dùng requests ở đây:
        # print("\nĐang thử tải từ GitHub (placeholder)...")
        # GITHUB_URL = "https://raw.githubusercontent.com/username/repo/main/data/train.parquet"
        # ... logic tải file ...

if __name__ == "__main__":
    download_data()
