import os
from src.model import create_ann_model

def train_dl_model(X_train_scaled, y_train):
    print("==========================================================")
    print("           STEP 3: TRAINING DEEP LEARNING (ANN)           ")
    print("==========================================================\n")
    
    # 1. Mengambil dimensi input dari jumlah fitur
    input_dim = X_train_scaled.shape[1]
    
    # 2. Membangun model
    model = create_ann_model(input_dim)
    
    print("\n[INFO] --- Ringkasan Arsitektur Model ---")
    model.summary()
    
    # 3. Proses Training
    print("\n[INFO] Memulai proses training (100 Epochs, Validation Split 20%)...")
    history = model.fit(
        X_train_scaled, 
        y_train, 
        epochs=100, 
        validation_split=0.2, 
        verbose=1
    )
    
    # 4. Menyimpan Model ke folder assets
    os.makedirs('assets', exist_ok=True)
    model_path = 'assets/ann_model.keras'
    model.save(model_path)
    
    print(f"\n[INFO] Training Selesai! Otak AI (Model) disimpan di: '{model_path}'\n")
    
    return model, history

if __name__ == '__main__':
    pass