import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test_scaled, y_test):
    print("==========================================================")
    print("      STEP 4: EVALUASI DEEP LEARNING (ANN) & METRIK       ")
    print("==========================================================\n")
    
    # =====================================================================
    # 1. Evaluasi Loss & Akurasi (Bawaan Keras)
    # =====================================================================
    print("[INFO] Melakukan evaluasi model pada data testing...")
    # verbose=0 agar tidak mengeluarkan progress bar panjang saat evaluasi
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\n[INFO] Akurasi di data uji: {acc * 100:.2f}%")
    print(f"[INFO] Loss di data uji   : {loss:.4f}\n")

    # =====================================================================
    # 2. Prediksi Kelas
    # =====================================================================
    print("[INFO] Melakukan prediksi pada data testing...")
    y_pred_prob = model.predict(X_test_scaled)
    
    # Threshold 0.5 untuk output Sigmoid (Ubah probabilitas menjadi kelas 0 / 1)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # =====================================================================
    # 3. Laporan Klasifikasi
    # =====================================================================
    print(" CLASSIFICATION REPORT (DEEP LEARNING) ".center(55, '='))
    print(classification_report(y_test, y_pred))
    print("=" * 55 + "\n")

    # =====================================================================
    # 4. Visualisasi Confusion Matrix
    # =====================================================================
    print("[INFO] Membuat visualisasi Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Deep Learning (ANN)')
    plt.tight_layout()

    # Simpan Gambar
    os.makedirs('assets', exist_ok=True)
    plot_path = 'assets/confusion_matrix_dl.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[INFO] Grafik Confusion Matrix berhasil disimpan di: '{plot_path}'\n")
    print("==========================================================")
    print("          PIPELINE DEEP LEARNING TELAH SELESAI!           ")
    print("==========================================================\n")

if __name__ == '__main__':
    pass