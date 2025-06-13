from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Activation, Input, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from AttLayer import AttLayer

class LSTMAtt:
    def __init__(self, hidden_node, input_shape, output_width):
        self.model = Sequential([
            Input(shape=input_shape),
            # Bidirectional(LSTM(hidden_node, return_sequences=True)),
            LSTM(hidden_node, return_sequences=True),
            Dropout(0.3),
            AttLayer(hidden_node),
            Dropout(0.3),
            Dense(output_width),
            BatchNormalization(),
            Activation('gelu'),
            Dense(32),
            BatchNormalization(),
            Activation('gelu'),
            Dense(3, activation='softmax')
        ])

        # Adam optimizer
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

        # Sử dụng AdamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0003,
        weight_decay=0.0008)

        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.compile(loss="categorical_hinge", optimizer=optimizer, metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        # Huấn luyện mô hình
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        # Dự đoán
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

        return accuracy, precision, recall, f1, conf_matrix, y_pred

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):

        # ModelCheckpoint: lưu trọng số của mô hình tốt nhất trong quá trình huấn luyện
        checkpoint_filepath = '/content/drive/MyDrive/Drowsiness-Detection/model_1.weights.h5'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1)

        # Huấn luyện mô hình
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            callbacks=[model_checkpoint]
        )
        return history


    def plot_training_history(self, history):
        """
        Plot training & validation loss and accuracy
        Args:
            history: History object returned by model.fit()
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training & validation loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Plot training & validation accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def save(self, filepath="/content/drive/MyDrive/Drowsiness-Detection/model.keras"):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath, custom_objects={'AttLayer': AttLayer})

    def plot_evaluation_metrics(self, x_test, y_test, history):
        # Plot training history
        self.plot_training_history(history)

        # Predict and convert to class labels
        y_pred = self.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Alert', 'Drowsiness', 'Yawn'],
                    yticklabels=['Alert', 'Drowsiness', 'Yawn'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Evaluation metrics
        accuracy, precision, recall, f1, conf_matrix, y_pred = self.evaluate(x_test, y_test)

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Confusion Matrix:\n{conf_matrix}")