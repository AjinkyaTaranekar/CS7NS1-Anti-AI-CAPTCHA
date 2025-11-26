import os
import numpy as np
from tensorflow.keras import layers, models
from emnist import extract_training_samples, extract_test_samples

# ============================================================
# 1. DATA PREPARATION / PREPARAZIONE DATI
# ============================================================

print("Loading EMNIST data (Balanced split)...")

# Load EMNIST 'balanced' (47 classes: 0-9, A-Z, and some a-z)
# Carica EMNIST 'balanced' (47 classi: 0-9, A-Z e alcune a-z)
train_images, train_labels = extract_training_samples('balanced')
test_images, test_labels = extract_test_samples('balanced')

# EMNIST images are rotated and flipped by default. We need to fix them.
# Le immagini EMNIST sono ruotate e specchiate di default. Dobbiamo correggerle.
train_images = np.transpose(train_images, (0, 2, 1))
test_images = np.transpose(test_images, (0, 2, 1))

# Normalize pixel values (0-1)
# Normalizza i valori dei pixel (0-1)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape for CNN: (Batch, Height, Width, Channel)
# Ridimensiona per la CNN: (Batch, Altezza, Larghezza, Canale)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Dictionary to map the prediction (0-46) to the actual character
# Dizionario per mappare la predizione (0-46) al carattere reale
label_map = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z',
    36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'
}

print(f"Data loaded. Training set size: {train_images.shape[0]}")

# ============================================================
# 2. MODEL ARCHITECTURE / ARCHITETTURA DEL MODELLO
# ============================================================

model = models.Sequential([
    # Conv Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Layer 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flattening and Dense Layers
    # Appiattimento e Livelli Densi
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Helps prevent overfitting / Aiuta a prevenire l'overfitting
    
    # Output Layer: 47 neurons for 47 classes
    # Strato di Output: 47 neuroni per 47 classi
    layers.Dense(47, activation='softmax')
])

# Compile the model
# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================================
# 3. TRAINING / ADDESTRAMENTO
# ============================================================

print("\nStarting training...")

# Train for 10 epochs
# Addestra per 10 epoche
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

print("Training finished.")

# ============================================================
# 4. EVALUATION / VALUTAZIONE
# ============================================================

print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# ============================================================
# 6. SAVE MODEL / SALVATAGGIO MODELLO
# ============================================================

model_filename = 'emnist_model.h5'
model.save(model_filename)
print(f"\nModel saved successfully as '{model_filename}'")