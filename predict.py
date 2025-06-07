import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
class SkinDiseasePredictor:
    def __init__(self, model_path='skin_disease_model.keras', img_size=(224, 224)):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.class_names = ['Eczema', 'Normal']
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            return img
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    def predict_single_image(self, image_path):
        """Make prediction for a single image"""
        img = self.preprocess_image(image_path)
        if img is None:
            return None, None
        img_batch = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img_batch, verbose=0)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        return predicted_class, confidence
    def predict_batch(self, image_folder):
        """Make predictions for all images in a folder"""
        results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        if not image_files:
            print(f"No images found in {image_folder}")
            return results
        print(f"\nProcessing {len(image_files)} images...")
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            predicted_class, confidence = self.predict_single_image(img_path)
            if predicted_class is not None:
                results.append({
                    'image': img_file,
                    'prediction': predicted_class,
                    'confidence': confidence
                })
                print(f"\nImage: {img_file}")
                print(f"Prediction: {predicted_class}")
                print(f"Confidence: {confidence:.2f}%")
        return results
    def visualize_predictions(self, image_folder, results, output_folder='predictions'):
        """Visualize predictions with images and confidence scores"""
        if not results:
            return
        os.makedirs(output_folder, exist_ok=True)
        n_images = len(results)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        plt.figure(figsize=(15, 5 * n_rows))
        for idx, result in enumerate(results, 1):
            img_path = os.path.join(image_folder, result['image'])
            img = self.preprocess_image(img_path)
            if img is not None:
                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(img)
                plt.title(f"{result['prediction']}\nConfidence: {result['confidence']:.2f}%")
                plt.axis('off')
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_folder, f'predictions_{timestamp}.png'))
        plt.close()
def main():
    predictor = SkinDiseasePredictor()
    image_folder = input("Enter the path to the folder containing images to predict: ").strip()
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' does not exist.")
        return
    results = predictor.predict_batch(image_folder)
    if results:
        predictor.visualize_predictions(image_folder, results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'prediction_results_{timestamp}.csv'
        with open(output_file, 'w') as f:
            f.write("Image,Prediction,Confidence\n")
            for result in results:
                f.write(f"{result['image']},{result['prediction']},{result['confidence']:.2f}\n")
        print(f"\nResults saved to {output_file}")
        print(f"Visualization saved in 'predictions' folder")
if __name__ == "__main__":
    main() 