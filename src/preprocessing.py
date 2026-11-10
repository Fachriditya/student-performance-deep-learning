import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(filepath='data/raw/BlaBla.csv', output_path='data/processed/cleaned_data.csv'):
    print("==========================================================")
    print("       STEP 1: DATA PREPROCESSING & ENCODING UMUR         ")
    print("==========================================================\n")
    
    # 1. Load Dataset
    try:
        df = pd.read_csv(filepath, sep=',')
    except Exception:
        df = pd.read_csv(filepath)

    # Membersihkan spasi ekstra pada seluruh nama kolom (Best Practice)
    df.columns = df.columns.str.strip()
    print(f"[INFO] Kolom yang terdeteksi di dataset: {df.columns.tolist()}")

    # =====================================================================
    # 2. ENCODING KOLOM UMUR (Sesuai Modul Praktikum)
    # =====================================================================
    if 'Umur' in df.columns:
        print("[INFO] Melakukan custom encoding pada kolom 'Umur'...")
        conditions = [
            (df['Umur'] <= 20),
            (df['Umur'] >= 21) & (df['Umur'] <= 30),
            (df['Umur'] >= 31) & (df['Umur'] <= 40),
            (df['Umur'] >= 41) & (df['Umur'] <= 50),
            (df['Umur'] > 50)
        ]
        choices = [1, 2, 3, 4, 5]
        # np.select akan mengaplikasikan logic di atas, default=1 sebagai fallback
        df['Umur'] = np.select(conditions, choices, default=1)
    else:
        print("[WARN] Kolom 'Umur' tidak ditemukan dalam dataset. Pastikan penamaan kolom benar.")

    # =====================================================================
    # 3. HANDLING MISSING VALUES
    # =====================================================================
    print("[INFO] Mengecek dan menangani Missing Values...")
    
    # Imputasi numerik dengan median (tahan terhadap outlier)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Imputasi kategorikal dengan modus
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # =====================================================================
    # 4. HANDLING OUTLIER (Metode IQR Capping)
    # =====================================================================
    print("[INFO] Menangani Outliers pada kolom numerik (Metode IQR Capping)...")
    # Kita tidak melakukan capping pada target (N) dan kolom Umur yang baru saja di-encode
    exclude_from_capping = ['Umur', 'N', 'target'] 
    
    for col in numeric_cols:
        if col not in exclude_from_capping:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Capping nilai yang melewati batas (winsorization)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # =====================================================================
    # 5. ENCODING SISA KOLOM KATEGORIKAL
    # =====================================================================
    print("[INFO] Melakukan Label Encoding pada sisa kolom kategorikal...")
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # =====================================================================
    # 6. EKSPOR DATA BERSIH
    # =====================================================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[INFO] Preprocessing selesai. Data bersih disimpan di: '{output_path}'\n")
    
    return df

if __name__ == '__main__':
    # Untuk uji coba lokal modul ini
    run_preprocessing()