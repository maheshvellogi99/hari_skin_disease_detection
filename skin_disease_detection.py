import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from datetime import datetime
class SkinDiseaseDetector:
    def __init__(self, dataset_path="dataset", img_size=(224, 224), batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = None
    def create_data_generators(self):
        """Create data generators with augmentation for training and validation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        self.val_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )    
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"Classes detected: {self.class_names}")
    def build_model(self):
        """Build and compile the CNN model"""
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(self.model.summary())
    def train_model(self, epochs=50):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        return history
    def evaluate_model(self):
        """Evaluate the model and generate reports"""
        loss, accuracy = self.model.evaluate(self.val_generator)
        print(f"\nValidation Accuracy: {accuracy:.2f}")
        print(f"Validation Loss: {loss:.2f}")
        predictions = self.model.predict(self.val_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.val_generator.classes
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    def predict_image(self, image_path):
        """Predict the class of a single image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    def save_model(self, model_path='skin_disease_model.keras'):
        """Save the trained model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    def load_saved_model(self, model_path='skin_disease_model.keras'):
        """Load a saved model"""
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
def plot_training_history(history):
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='train_loss', color='tomato')
    plt.plot(history.history['val_loss'], label='val_loss', color='skyblue')
    plt.plot(history.history['accuracy'], label='train_acc', color='slateblue')
    plt.plot(history.history['val_accuracy'], label='val_acc', color='gray')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
def main():
    detector = SkinDiseaseDetector(
        dataset_path="dataset",
        img_size=(224, 224),
        batch_size=32
    )
    detector.create_data_generators()
    detector.build_model()
    history = detector.train_model(epochs=20)
    detector.evaluate_model()
    detector.save_model()
    plot_training_history(history)
if __name__ == "__main__":
    main() 