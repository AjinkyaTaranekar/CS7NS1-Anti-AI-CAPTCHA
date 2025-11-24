import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# CONFIGURATION / CONFIGURAZIONE
# ==========================================
BATCH_SIZE = 32
IMG_WIDTH = 128
IMG_HEIGHT = 64
EPOCHS = 50
MAX_LENGTH = 5
DATA_DIR = "./dataset"
CHECKPOINT_DIR = "./checkpoints"

# Crea cartella checkpoint
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Vocabolario
characters = sorted(list("abcdefghijklmnopqrstuvwxyz0123456789"))
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

print(f"Found {len(characters)} unique characters.")

# ==========================================
# DATA LOADER
# ==========================================
def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    
    # --- USIAMO 3 CANALI (RGB) INVECE DI 1 ---
    # --- USE 3 CHANNELS (RGB) INSTEAD OF 1 ---
    img = tf.io.decode_png(img, channels=3) 
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}

images = sorted(list(map(str, list(Path(DATA_DIR).glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split("_")[0] for img in images]

print(f"Dataset size: {len(images)} images.")

if len(images) == 0:
    print("ERROR: No images found in 'dataset' folder!")
    exit()

x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_models():
    # --- INPUT SHAPE HA 3 CANALI ---
    # --- INPUT SHAPE HAS 3 CHANNELS ---
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    output_loss = CTCLayer(name="ctc_loss")(labels, x)
    training_model = keras.models.Model(
        inputs=[input_img, labels], outputs=output_loss, name="ocr_training"
    )
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    training_model.compile(optimizer=opt)

    prediction_model = keras.models.Model(
        inputs=input_img, outputs=x, name="ocr_prediction"
    )

    return training_model, prediction_model

# ==========================================
# CALLBACKS
# ==========================================
class CheckpointSaver(keras.callbacks.Callback):
    def __init__(self, pred_model, interval=5):
        self.pred_model = pred_model
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            filename = f"{CHECKPOINT_DIR}/model_epoch_{epoch + 1:02d}.h5"
            print(f"\nSaving checkpoint: {filename}")
            self.pred_model.save(filename)

model, prediction_model = build_models()
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

checkpoint_saver = CheckpointSaver(prediction_model, interval=5)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1
)

print("Starting training (RGB Mode)...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint_saver, reduce_lr],
)

prediction_model.save("best_model.h5")
print("Best model saved to 'best_model.h5'")