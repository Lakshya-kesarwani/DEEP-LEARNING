# train_model.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import CSVLogger, Callback
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from models import all_models
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# === Paths and Setup ===
DATA_DIR = 'dataset/images'
LOGS_DIR = 'logs'
MODELS_DIR = 'saved_models'
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(img_size):
    X, y = [], []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jpg'):
            try:
                parts = filename.split('_')
                if len(parts) < 4:
                    raise ValueError("Filename does not contain enough parts")

                x = float(parts[1])        # x coordinate
                y_coord = float(parts[2])  # y coordinate

                path = os.path.join(DATA_DIR, filename)
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Image {filename} could not be loaded.")
                img = cv2.resize(img, (img_size[0], img_size[1]))

                # Normalize coordinates to [-1, 1]
                X.append(img)
                y.append([(x / 177) - 1, (y_coord / 396) - 1])

            except Exception as e:
                print(f"Skipping {filename}: {e}")

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.float32)
    return X, y


# === R² Metric ===
def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# === Custom Callback ===
class PrintValAccuracy(Callback):
    def __init__(self, models_dir, name=''):
        super().__init__()
        self.models_dir = models_dir
        self.previous_mae = float('inf')
        self.name = name
        self.best_model_path = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_mae = logs.get('val_mae')
        val_loss = logs.get('val_loss')

        if val_mae is not None and val_mae < self.previous_mae and val_mae <= 0.1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"model_{self.name}_epoch_{epoch + 1}_val_mae_{val_mae:.4f}_{timestamp}.h5"
            model_path = os.path.join(self.models_dir, model_name)
            os.makedirs(self.models_dir, exist_ok=True)
            self.model.save(model_path)
            self.previous_mae = val_mae
            self.best_model_path = model_path
            print(f"✅ Model saved: {model_name}")

        if val_mae is not None and val_loss is not None:
            print(f"📊 Epoch {epoch + 1}: val_mae = {val_mae:.4f} | val_loss = {val_loss:.4f}")

# === Train and Evaluate (with Adam Optimizer and Enhanced Plotting) ===
def train_and_evaluate(model_fn, model_name, img_size, X_train_val, X_test, y_train_val, y_test):
    model = model_fn(input_shape=(img_size[0], img_size[1], 3))

    # Exponential decay learning rate for Adam
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )

    # Adam optimizer with decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    log_path = os.path.join(LOGS_DIR, f'{model_name}_{img_size}_tanh.csv')
    csv_logger = CSVLogger(log_path)
    print_callback = PrintValAccuracy(os.path.join(MODELS_DIR, model_name), name=model_name)
    tqdm_callback = TqdmCallback(verbose=1)

    # 80/20 train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        verbose=0,
        callbacks=[csv_logger, print_callback, tqdm_callback]
    )

    # Load best model or fallback to final model
    if print_callback.best_model_path:
        print(f"🔁 Loading best model from: {print_callback.best_model_path}")
        best_model = load_model(print_callback.best_model_path, compile=False)
        best_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        print("⚠️ No model saved during training, using final model.")
        best_model = model

    # Evaluate
    loss, mae = best_model.evaluate(X_test, y_test, verbose=0)
    y_pred = best_model.predict(X_test)
    r2 = calculate_r2(y_test, y_pred)

    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(MODELS_DIR, f'tanh_{model_name}_{img_size[0]}_{img_size[1]}_{timestamp}.h5')
    best_model.save(final_model_path)

    # Plot both MSE and MAE
    plt.figure()
    plt.plot(history.history['loss'], label='Train MSE')
    plt.plot(history.history['val_loss'], label='Val MSE')
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{model_name} - {img_size[0]}x{img_size[1]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(LOGS_DIR, f'{model_name}_{img_size}_loss_mae.png')
    plt.savefig(plot_path)
    plt.close()

    return loss, mae, r2, best_model

# === Comparison Plot ===
def plot_results(results):
    mse_values = []
    mae_values = []
    labels = []

    for name, img_size, mse, mae in results:
        labels.append(f"{name}\n({img_size[0]}x{img_size[1]})")
        mse_values.append(mse)
        mae_values.append(mae)

    x = np.arange(len(labels))
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    plt.bar(x - bar_width / 2, mse_values, width=bar_width, label='MSE', color='skyblue')
    plt.bar(x + bar_width / 2, mae_values, width=bar_width, label='MAE', color='orange')

    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Model + Image Size', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, 'model_comparison.png'))
    plt.clf()
    print(f"\n📊 Comparison graph saved at: logs/model_comparison.png")

# === Main ===
def main():
    image_sizes = [[180,320]]  # You can add more here
    results = []

    for img_size in image_sizes:
        print(f"\n=== Processing image size: {img_size} ===")
        X, y = load_data(img_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model_fn in all_models:
            try:
                print(f"\n🚀 Training {name} at image size {img_size}...")
                loss, mae, r2, trained_model = train_and_evaluate(
                    model_fn, name.replace(' ', '_'), img_size, X_train, X_test, y_train, y_test)
                results.append((name, img_size, loss, mae, r2))
                print(f"✅ {name} | MSE: {loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                results.append((name, img_size, float('inf'), float('inf'), float('-inf')))

    best_model = min(results, key=lambda x: x[2])
    print("\n=== Final Results ===")
    for name, img_size, mse, mae, r2 in results:
        print(f"Model: {name}, Image Size: {img_size}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    csv_path = os.path.join(LOGS_DIR, 'final_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Model,Image Size,MSE,MAE,R2\n")
        for name, img_size, mse, mae, r2 in results:
            f.write(f"{name},{img_size},{mse:.4f},{mae:.4f},{r2:.4f}\n")

    print(f"\n🏆 Best model: {best_model[0]} at {best_model[1]} | MSE: {best_model[2]:.4f}, MAE: {best_model[3]:.4f}, R2: {best_model[4]:.4f}")
    print(f"📁 Results saved to: {csv_path}")

    plot_results([(name, img_size, mse, mae) for name, img_size, mse, mae, _ in results])

if __name__ == "__main__":
    # run on gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("✅ GPU is available and will be used.")
        except RuntimeError as e:
            print(f"❌ Error setting up GPU: {e}")
    else:
        print("⚠️ No GPU found. The code will run on CPU.")
    main()
    print("\n✅ Training complete and results saved.")