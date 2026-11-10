import sys
import os

# Agar folder 'src' dan isinya terbaca oleh sistem Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import run_preprocessing
from src.splitter import run_splitting
from src.train import train_dl_model
from src.evaluate import evaluate_model

if __name__ == '__main__':
    print("[INFO] Memulai Orkestrasi Pipeline Klasifikasi Deep Learning (ANN)...")
    
    # 1. Preprocessing Data
    # Pastikan file CSV raw ada di folder data/raw/
    df_cleaned = run_preprocessing(
        filepath='data/raw/BlaBla.csv', 
        output_path='data/processed/cleaned_data.csv'
    )
    
    # 2. Splitting & Scaling
    # Catatan: Kolom target sesuai instruksi modul adalah 'N'
    X_train_s, X_test_s, y_train, y_test = run_splitting(df_cleaned, target_column='N')
    
    # 3. Membangun dan Melatih Model (100 Epochs)
    model, history = train_dl_model(X_train_s, y_train)
    
    # 4. Mengevaluasi Model & Export Visualisasi
    evaluate_model(model, X_test_s, y_test)