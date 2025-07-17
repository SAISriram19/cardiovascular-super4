import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# --- Configuration ---
# Paths to the preprocessed PTB-XL dataset files
PTB_XL_FEATURE_PATH = "/kaggle/input/preprocessed/Feature"
PTB_XL_LABEL_PATH = "/kaggle/input/preprocessed/Label/label.npy"

# Constants
TARGET_LENGTH = 187
MODEL_OUTPUT_NAME = 'ptb_xl_model.keras'
SCALER_OUTPUT_NAME = 'ptb_xl_scaler.pkl'

# A specific mapping for the labels in this dataset
PTB_XL_LABELS = {
    "Normal": 0,
    "Myocardial_Infarction": 1,
    "ST_T_Change": 2,
    "Conduction_Disturbance": 3,
    "Hypertrophy": 4,
    "Other": 5
}

# --- Data Loading ---
def load_ptb_xl_data(feature_path, label_path):
    """Loads the preprocessed PTB-XL dataset from .npy files."""
    print("Loading preprocessed PTB-XL data...")
    try:
        feature_files = sorted(glob(os.path.join(feature_path, 'feature_*.npy')))
        if not feature_files:
            raise FileNotFoundError(f"No feature files found at {feature_path}")
            
        valid_features = []
        for f in tqdm(feature_files, desc="Loading features"):
            feature = np.load(f, allow_pickle=True)
            if feature.ndim == 1 and len(feature) == TARGET_LENGTH:
                valid_features.append(feature)
        
        X = np.array(valid_features, dtype=np.float32)
        raw_labels = np.load(label_path, allow_pickle=True)
        
        # Align data and labels
        min_length = min(len(X), len(raw_labels))
        X = X[:min_length]
        raw_labels = raw_labels[:min_length]

        # Map raw labels to our defined labels
        y = []
        for label_arr in raw_labels:
            original_label = label_arr[0] if isinstance(label_arr, (list, np.ndarray)) and len(label_arr) > 0 else 0
            if original_label == 0: y.append(PTB_XL_LABELS["Normal"])
            elif original_label == 1: y.append(PTB_XL_LABELS["Myocardial_Infarction"])
            elif original_label == 2: y.append(PTB_XL_LABELS["ST_T_Change"])
            elif original_label == 3: y.append(PTB_XL_LABELS["Conduction_Disturbance"])
            elif original_label == 4: y.append(PTB_XL_LABELS["Hypertrophy"])
            else: y.append(PTB_XL_LABELS["Other"])
        
        y = np.array(y)
        print(f"Successfully loaded data. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the Kaggle dataset is available at the specified paths.")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return np.array([]), np.array([])

# --- Model Architecture ---
def create_model(input_shape, num_classes):
    """Creates the CNN-LSTM model architecture."""
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(64, activation="tanh"),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Training Pipeline ---
def process_and_train():
    """Main function to run the data processing and model training pipeline."""
    # 1. Load Data
    X, y = load_ptb_xl_data(PTB_XL_FEATURE_PATH, PTB_XL_LABEL_PATH)
    if X.size == 0:
        print("Training aborted due to data loading failure.")
        return

    num_classes = len(np.unique(y))
    class_names = list(PTB_XL_LABELS.keys())

    # 2. Train-Validation Split
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale Data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # 4. Apply SMOTE to the training data
    print(f"\nOriginal training data shape: {x_train_scaled.shape}")
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)
    print(f"Resampled training data shape: {x_train_resampled.shape}")

    # 5. Reshape and One-Hot Encode
    x_train_reshaped = x_train_resampled.reshape(x_train_resampled.shape[0], TARGET_LENGTH, 1)
    x_val_reshaped = x_val_scaled.reshape(x_val_scaled.shape[0], TARGET_LENGTH, 1)
    y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=num_classes)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    # 6. Create and Train Model
    model = create_model(input_shape=(TARGET_LENGTH, 1), num_classes=num_classes)
    print("\nModel Summary:")
    print(model.summary())

    callbacks = [
        ModelCheckpoint(MODEL_OUTPUT_NAME, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    print("\nStarting model training...")
    history = model.fit(
        x_train_reshaped, y_train_onehot, 
        epochs=30, 
        batch_size=64, 
        validation_data=(x_val_reshaped, y_val_onehot), 
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluate and Save
    print("\nEvaluating final model on the validation set...")
    val_loss, val_acc = model.evaluate(x_val_reshaped, y_val_onehot)
    print(f"Validation Accuracy: {val_acc:.4f}")

    y_pred = model.predict(x_val_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nValidation Set Classification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=class_names, zero_division=0))

    # Save the scaler
    with open(SCALER_OUTPUT_NAME, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {MODEL_OUTPUT_NAME}")
    print(f"Scaler saved to {SCALER_OUTPUT_NAME}")

if __name__ == "__main__":
    process_and_train()
