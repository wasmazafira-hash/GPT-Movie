"""
Movie Success Prediction using Neural Networks

This script predicts movie success based on various features like budget, 
runtime, genre, director rating, and actor popularity.

Fixed version: Uses plt.savefig() instead of plt.show() for better compatibility
with headless environments and Jupyter notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# SECTION 1: DATA CREATION
# ============================================================================

try:
    print("=" * 80)
    print("SECTION 1: Creating Movie Dataset")
    print("=" * 80)

    np.random.seed(42)

    rows = 5000

    data = pd.DataFrame({
        'Budget': np.random.randint(20, 300, rows),
        'Runtime': np.random.randint(80, 180, rows),
        'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'SciFi', 'Horror'], rows),
        'Director_Rating': np.random.uniform(1, 10, rows),
        'Actor_Popularity': np.random.uniform(1, 10, rows),
        'Marketing_Budget': np.random.uniform(5, 100, rows),
        'Release_Month': np.random.uniform(1, 13, rows)
    })

    score = (
        data['Budget'] * 0.3 +
        data['Director_Rating'] * 10 +
        data['Actor_Popularity'] * 12 +
        data['Marketing_Budget'] * 1.5
    )

    data['Success'] = pd.cut(
        score,
        bins=[0, 120, 220, 1000],
        labels=[0, 1, 2]
    )

    data.to_csv('movie_success.csv', index=False)

    print("Dataset created successfully!")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())

    # ============================================================================
    # SECTION 2: DATA LOADING AND EXPLORATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 2: Loading and Exploring Data")
    print("=" * 80)

    data = pd.read_csv('movie_success.csv')

    print(f"\nDataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())

    # Plot success distribution
    sns.set()
    sns.countplot(x='Success', data=data)
    plt.title('Movie Success Distribution')
    plt.savefig('01_success_distribution.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 01_success_distribution.png")
    plt.close()

    # ============================================================================
    # SECTION 3: DATA PREPROCESSING
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 3: Data Preprocessing")
    print("=" * 80)

    # Encode categorical variable
    encoder = LabelEncoder()
    data['Genre'] = encoder.fit_transform(data['Genre'])
    print("Genre encoded successfully!")

    # Split features and target
    X = data.drop('Success', axis=1)
    y = data['Success']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Features scaled successfully!")

    # Convert target to categorical
    y = to_categorical(y, num_classes=3)
    print(f"Target converted to categorical shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # ============================================================================
    # SECTION 4: MODEL BUILDING
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 4: Building Neural Network Model")
    print("=" * 80)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    print("Model architecture created!")

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel compiled!")
    print("\nModel Summary:")
    model.summary()

    # ============================================================================
    # SECTION 5: MODEL TRAINING
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 5: Training Model")
    print("=" * 80)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=3
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\nModel training completed!")

    # ============================================================================
    # SECTION 6: TRAINING HISTORY VISUALIZATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 6: Visualizing Training History")
    print("=" * 80)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Train', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Train', 'Validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('02_training_history.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 02_training_history.png")
    plt.close()

    # ============================================================================
    # SECTION 7: MODEL EVALUATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 7: Model Evaluation")
    print("=" * 80)

    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(y_test, axis=1)

    print("Predictions generated!")

    # Confusion Matrix
    cm = confusion_matrix(actual_classes, pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Movie Success Prediction')
    plt.savefig('03_confusion_matrix.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 03_confusion_matrix.png")
    plt.close()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(actual_classes, pred_classes))

    # ============================================================================
    # SECTION 8: PREDICTION ON NEW DATA
    # ============================================================================

    print("\n" + "=" * 80)
    print("SECTION 8: Predicting Success for New Movie")
    print("=" * 80)

    new_movie = pd.DataFrame([{
        'Budget': 150,
        'Runtime': 130,
        'Genre': encoder.transform(['Action'])[0],
        'Director_Rating': 8.5,
        'Actor_Popularity': 9.0,
        'Marketing_Budget': 50,
        'Release_Month': 6
    }])

    print("New movie data:")
    print(new_movie)

    new_movie_scaled = scaler.transform(new_movie)
    prediction = model.predict(new_movie_scaled)
    result = np.argmax(prediction)

    success_labels = {
        0: 'Low Success',
        1: 'Medium Success',
        2: 'High Success'
    }

    print(f"\nPredicted Success Level: {success_labels[result]}")
    print(f"Confidence: {prediction[0][result]:.2%}")
    print("\nAll prediction probabilities:")
    print(f"  Low Success: {prediction[0][0]:.2%}")
    print(f"  Medium Success: {prediction[0][1]:.2%}")
    print(f"  High Success: {prediction[0][2]:.2%}")

    print("\n" + "=" * 80)
    print("✓ Analysis Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - movie_success.csv (dataset)")
    print("  - 01_success_distribution.png")
    print("  - 02_training_history.png")
    print("  - 03_confusion_matrix.png")

except Exception as e:
    print(f"\n❌ ERROR: {e}", file=sys.stderr)
    print("\nFull traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
