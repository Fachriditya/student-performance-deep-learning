import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_splitting(df, target_column='N'):
    print("==========================================================")
    print("          STEP 2: SPLITTING & FEATURE SCALING             ")
    print("==========================================================\n")
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"[INFO] Fitur Input (X) shape: {X.shape}")
    print(f"[INFO] Target Output (y) shape: {y.shape}")
    
    # Melakukan Splitting (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("[INFO] Splitting selesai. Data Training 80%, Data Testing 20%.")
    
    # Feature Scaling khusus untuk Deep Learning (ANN)
    print("[INFO] Melakukan Feature Scaling menggunakan StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("[INFO] Data siap dimasukkan ke dalam model Neural Network!\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    pass