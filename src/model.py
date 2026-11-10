from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_ann_model(input_dim):
    print("[INFO] Inisialisasi Arsitektur Sequential ANN...")
    
    # Bangun model klasifikasi biner sesuai spesifikasi
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Output binary (1 Neuron, Sigmoid)
    ])
    
    print("[INFO] Melakukan Compile Model (Optimizer: Adam, Loss: binary_crossentropy)...")
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    pass