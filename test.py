#!/usr/bin/env python3
"""
Enhanced Eye Disease Prediction Script
=====================================
Improved version with better debugging, preprocessing, and error handling.

Usage:
    1. Place your images in the 'input' folder
    2. Run: python predict_images_enhanced.py
    3. Results will be saved in 'output' folder and printed to console

Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import warnings
import cv2

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EnhancedEyeDiseasePredictor:
    def __init__(self, model_path="eye_disease_efficientnet_best.h5"):
        """
        Initialize the Enhanced Eye Disease Predictor with better preprocessing
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.model_is_healthy = False
        
        # Disease classes - same as in training
        self.disease_classes = [
            'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS',
            'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
            'VB', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL', 'VH', 'MHL', 'RP',
            'CRS', 'EDN', 'RPEC', 'MH', 'CN', 'CORD', 'SRF', 'DM', 'MYD', 'SMH',
            'RTPE', 'HR', 'CRAO', 'TD', 'CME'
        ]
        
        # Disease class descriptions for better understanding
        self.disease_descriptions = {
            'DR': 'Diabetic Retinopathy',
            'ARMD': 'Age-Related Macular Degeneration',
            'MH': 'Macular Hole',
            'DN': 'Diabetic Nephropathy',
            'MYA': 'Myopia',
            'BRVO': 'Branch Retinal Vein Occlusion',
            'TSLN': 'Tessellation',
            'ERM': 'Epiretinal Membrane',
            'LS': 'Laser Scars',
            'MS': 'Macular Scar',
            'CSR': 'Central Serous Retinopathy',
            'ODC': 'Optic Disc Cupping',
            'CRVO': 'Central Retinal Vein Occlusion',
            'TV': 'Tortuous Vessels',
            'AH': 'Asteroid Hyalosis',
            'ODP': 'Optic Disc Pallor',
            'ODE': 'Optic Disc Edema',
            'ST': 'Optociliary Shunt',
            'AION': 'Anterior Ischemic Optic Neuropathy',
            'PT': 'Parafoveal Telangiectasia',
            'VB': 'Vitreous Bands',
            'MCA': 'Macular Atrophy',
            'VS': 'Vitreous Syneresis',
            'BRAO': 'Branch Retinal Artery Occlusion',
            'PLQ': 'Plaque',
            'HPED': 'Hemorrhagic Pigment Epithelial Detachment',
            'CL': 'Collaterals',
            'VH': 'Vitreous Hemorrhage',
            'MHL': 'Macular Hard Exudates',
            'RP': 'Retinitis Pigmentosa',
            'CRS': 'Chorioretinal Scar',
            'EDN': 'Exudation',
            'RPEC': 'Retinal Pigment Epithelium Changes',
            'CN': 'Cotton Wool Spots',
            'CORD': 'Chorioretinal Dystrophy',
            'SRF': 'Subretinal Fluid',
            'DM': 'Diastolic Murmur',
            'MYD': 'Mydriasis',
            'SMH': 'Submacular Hemorrhage',
            'RTPE': 'Retinal Pigment Epithelium',
            'HR': 'Hard Exudates',
            'CRAO': 'Central Retinal Artery Occlusion',
            'TD': 'Tilted Disc',
            'CME': 'Cystoid Macular Edema'
        }
        
        self.img_size = 224  # EfficientNet input size
        self.input_folder = "input"
        self.output_folder = "output"
        
        self._setup_folders()
        self._load_and_validate_model()
    
    def _setup_folders(self):
        """Create input and output folders if they don't exist"""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"[INFO] Input folder: {self.input_folder}")
        print(f"[INFO] Output folder: {self.output_folder}")
    
    def _load_and_validate_model(self):
        """Load and validate the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[ERROR] Model file not found: {self.model_path}")
        
        try:
            print(f"[LOADING] Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            print(f"[SUCCESS] Model loaded successfully!")
            print(f"[INFO] Model input shape: {self.model.input_shape}")
            print(f"[INFO] Model output shape: {self.model.output_shape}")
            
            # Validate model health
            self._validate_model_health()
            
        except Exception as e:
            raise Exception(f"[ERROR] Error loading model: {str(e)}")
    
    def _validate_model_health(self):
        """Check if the model has proper weights and can make varied predictions"""
        print(f"[VALIDATION] Checking model health...")
        
        # Check weights
        weights = self.model.get_weights()
        print(f"[INFO] Model has {len(weights)} weight arrays")
        
        # Check if weights are not all zeros
        non_zero_weights = 0
        total_weights = 0
        for weight_array in weights:
            if len(weight_array.shape) > 0:  # Skip empty arrays
                total_weights += 1
                if np.any(weight_array != 0):
                    non_zero_weights += 1
        
        print(f"[INFO] Non-zero weight arrays: {non_zero_weights}/{total_weights}")
        
        # Test with different inputs
        print(f"[VALIDATION] Testing model with different inputs...")
        
        # Test 1: All zeros
        zero_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        pred_zero = self.model.predict(zero_input, verbose=0)[0]
        
        # Test 2: All ones
        ones_input = np.ones((1, 224, 224, 3), dtype=np.float32)
        pred_ones = self.model.predict(ones_input, verbose=0)[0]
        
        # Test 3: Random input
        random_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred_random = self.model.predict(random_input, verbose=0)[0]
        
        # Check if predictions are different
        diff_zero_ones = np.mean(np.abs(pred_zero - pred_ones))
        diff_zero_random = np.mean(np.abs(pred_zero - pred_random))
        diff_ones_random = np.mean(np.abs(pred_ones - pred_random))
        
        print(f"[INFO] Prediction differences:")
        print(f"  Zero vs Ones: {diff_zero_ones:.6f}")
        print(f"  Zero vs Random: {diff_zero_random:.6f}")
        print(f"  Ones vs Random: {diff_ones_random:.6f}")
        
        # Model is healthy if predictions vary significantly
        avg_diff = (diff_zero_ones + diff_zero_random + diff_ones_random) / 3
        self.model_is_healthy = avg_diff > 0.001  # Threshold for variation
        
        if self.model_is_healthy:
            print(f"[SUCCESS] Model appears healthy - predictions vary with input")
        else:
            print(f"[WARNING] Model may have issues - predictions don't vary much")
            print(f"[WARNING] This could indicate:")
            print(f"  • Model weights are frozen or corrupted")
            print(f"  • Model architecture issues")
            print(f"  • Model wasn't trained properly")
        
        print(f"[INFO] Model validation complete\n")
    
    def _get_image_files(self):
        """Get all image files from input folder"""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        if not os.path.exists(self.input_folder):
            print(f"[ERROR] Input folder '{self.input_folder}' not found!")
            return image_files
        
        for file in os.listdir(self.input_folder):
            if any(file.lower().endswith(ext) for ext in supported_formats):
                image_files.append(file)
        
        return sorted(image_files)
    
    def _preprocess_image_method1(self, image_path):
        """
        Method 1: Basic preprocessing (original method)
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def _preprocess_image_method2(self, image_path):
        """
        Method 2: EfficientNet specific preprocessing
        """
        img = image.load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # EfficientNet preprocessing
        return img_array
    
    def _preprocess_image_method3(self, image_path):
        """
        Method 3: OpenCV with enhanced preprocessing
        """
        # Read image with OpenCV
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        img = cv2.merge((l_channel, a_channel, b_channel))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        
        # Normalize
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _preprocess_image(self, image_path, method=1):
        """
        Preprocess image using specified method
        
        Args:
            image_path (str): Path to the image file
            method (int): Preprocessing method (1, 2, or 3)
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            if method == 1:
                return self._preprocess_image_method1(image_path)
            elif method == 2:
                return self._preprocess_image_method2(image_path)
            elif method == 3:
                return self._preprocess_image_method3(image_path)
            else:
                raise ValueError(f"Invalid preprocessing method: {method}")
                
        except Exception as e:
            raise Exception(f"Error preprocessing image {image_path}: {str(e)}")
    
    def predict_single_image(self, image_path, top_k=5, threshold=0.5, preprocessing_method=1):
        """
        Predict diseases for a single image with multiple preprocessing methods
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return
            threshold (float): Probability threshold for positive prediction
            preprocessing_method (int): Which preprocessing method to use
            
        Returns:
            dict: Prediction results
        """
        results = {}
        
        # Try different preprocessing methods
        methods_to_try = [preprocessing_method] if preprocessing_method in [1, 2, 3] else [1, 2, 3]
        
        for method in methods_to_try:
            try:
                # Preprocess image
                img_array = self._preprocess_image(image_path, method)
                
                # Make prediction
                predictions = self.model.predict(img_array, verbose=0)[0]
                
                # Create results for this method
                method_result = {
                    'preprocessing_method': method,
                    'method_name': f"Method {method}",
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'all_predictions': {},
                    'top_predictions': [],
                    'positive_predictions': [],
                    'max_probability': float(np.max(predictions)),
                    'prediction_summary': {},
                    'model_health_warning': not self.model_is_healthy
                }
                
                # Store all predictions
                for i, (disease, prob) in enumerate(zip(self.disease_classes, predictions)):
                    method_result['all_predictions'][disease] = {
                        'probability': float(prob),
                        'description': self.disease_descriptions.get(disease, disease)
                    }
                
                # Get top K predictions
                top_indices = np.argsort(predictions)[::-1][:top_k]
                for idx in top_indices:
                    disease = self.disease_classes[idx]
                    prob = predictions[idx]
                    method_result['top_predictions'].append({
                        'disease': disease,
                        'probability': float(prob),
                        'description': self.disease_descriptions.get(disease, disease)
                    })
                
                # Get positive predictions (above threshold)
                positive_indices = np.where(predictions >= threshold)[0]
                for idx in positive_indices:
                    disease = self.disease_classes[idx]
                    prob = predictions[idx]
                    method_result['positive_predictions'].append({
                        'disease': disease,
                        'probability': float(prob),
                        'description': self.disease_descriptions.get(disease, disease)
                    })
                
                # Create summary
                method_result['prediction_summary'] = {
                    'total_diseases_detected': len(positive_indices),
                    'highest_probability_disease': self.disease_classes[np.argmax(predictions)],
                    'confidence_level': 'High' if np.max(predictions) > 0.8 else 'Medium' if np.max(predictions) > 0.5 else 'Low',
                    'prediction_entropy': float(-np.sum(predictions * np.log(predictions + 1e-10))),  # Measure of uncertainty
                    'prediction_variance': float(np.var(predictions))  # Measure of spread
                }
                
                results[f'method_{method}'] = method_result
                
            except Exception as e:
                print(f"Error with preprocessing method {method}: {str(e)}")
                continue
        
        return results
    
    def predict_all_images(self, top_k=5, threshold=0.5, preprocessing_method=1, save_results=True):
        """
        Predict diseases for all images in the input folder
        
        Args:
            top_k (int): Number of top predictions to return per image
            threshold (float): Probability threshold for positive prediction
            preprocessing_method (int): Which preprocessing method to use (1, 2, 3, or 0 for all)
            save_results (bool): Whether to save results to files
            
        Returns:
            list: List of prediction results for all images
        """
        image_files = self._get_image_files()
        
        if not image_files:
            print(f"No image files found in '{self.input_folder}' folder!")
            print(f"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
            return []
        
        print(f"\nFound {len(image_files)} images to process...")
        
        all_results = []
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(self.input_folder, image_file)
            
            try:
                print(f"\nProcessing image {i}/{len(image_files)}: {image_file}")

                # Make prediction
                result = self.predict_single_image(image_path, top_k, threshold, preprocessing_method)
                all_results.append(result)
                
                # Print summary for each preprocessing method
                for method_key, method_result in result.items():
                    print(f"Method {method_result['method_name']} processed successfully!")
                    print(f"Diseases detected (>={threshold} threshold): {method_result['prediction_summary']['total_diseases_detected']}")
                    print(f"Highest probability: {method_result['prediction_summary']['highest_probability_disease']} "
                          f"({method_result['max_probability']:.4f})")
                    print(f"Confidence Level: {method_result['prediction_summary']['confidence_level']}")
                    
                    if method_result['model_health_warning']:
                        print(f"WARNING: Model health warning - predictions may not be reliable")

                    print(f"\nTop 3 Predictions with Confidence:")
                    for j, pred in enumerate(method_result['top_predictions'][:3], 1):
                        confidence_percent = pred['probability'] * 100
                        print(f"  {j}. {pred['disease']}: {pred['probability']:.4f} ({confidence_percent:.2f}%) - {pred['description']}")

                    # Show if ANY diseases were detected above a lower threshold
                    low_threshold = 0.1  # 10% threshold
                    diseases_above_low_threshold = [p for p in method_result['top_predictions'] if p['probability'] >= low_threshold]
                    if diseases_above_low_threshold:
                        print(f"\nDiseases above {low_threshold:.0%} threshold:")
                        for pred in diseases_above_low_threshold[:5]:
                            confidence_percent = pred['probability'] * 100
                            print(f"  • {pred['disease']}: {confidence_percent:.2f}% - {pred['description']}")
                    print("-" * 50)

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # Save results if requested
        if save_results and all_results:
            self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, all_results):
        """Save prediction results to files with enhanced formatting"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed JSON results
        json_path = os.path.join(self.output_folder, f'predictions_enhanced_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")
        
        # Save enhanced summary CSV
        csv_data = []
        for result in all_results:
            for method_key, method_result in result.items():
                row = {
                    'image_name': method_result['image_name'],
                    'preprocessing_method': method_result['preprocessing_method'],
                    'method_name': method_result['method_name'],
                    'timestamp': method_result['timestamp'],
                    'diseases_detected': method_result['prediction_summary']['total_diseases_detected'],
                    'top_disease': method_result['prediction_summary']['highest_probability_disease'],
                    'max_probability': method_result['max_probability'],
                    'confidence_level': method_result['prediction_summary']['confidence_level'],
                    'prediction_entropy': method_result['prediction_summary']['prediction_entropy'],
                    'prediction_variance': method_result['prediction_summary']['prediction_variance'],
                    'model_health_warning': method_result['model_health_warning']
                }
                
                # Add top 3 predictions
                for j, pred in enumerate(method_result['top_predictions'][:3]):
                    row[f'top_{j+1}_disease'] = pred['disease']
                    row[f'top_{j+1}_probability'] = pred['probability']
                    row[f'top_{j+1}_description'] = pred['description']
                
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_folder, f'predictions_enhanced_summary_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Enhanced summary CSV saved to: {csv_path}")
        
        # Save comprehensive text report
        txt_path = os.path.join(self.output_folder, f'predictions_enhanced_report_{timestamp}.txt')
        with open(txt_path, 'w') as f:
            f.write("Enhanced Eye Disease Prediction Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Processed: {len(all_results)}\n")
            f.write(f"Model Health Status: {'WARNING' if not self.model_is_healthy else 'GOOD'}\n\n")
            
            if not self.model_is_healthy:
                f.write("MODEL HEALTH WARNING:\n")
                f.write("The model may not be functioning correctly. Predictions may not be reliable.\n")
                f.write("Consider retraining or checking the model file.\n\n")
            
            for i, result in enumerate(all_results, 1):
                for method_key, method_result in result.items():
                    f.write(f"Image {i}: {method_result['image_name']} ({method_result['method_name']})\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Diseases Detected: {method_result['prediction_summary']['total_diseases_detected']}\n")
                    f.write(f"Confidence Level: {method_result['prediction_summary']['confidence_level']}\n")
                    f.write(f"Max Probability: {method_result['max_probability']:.3f}\n")
                    f.write(f"Prediction Entropy: {method_result['prediction_summary']['prediction_entropy']:.3f}\n")
                    f.write(f"Prediction Variance: {method_result['prediction_summary']['prediction_variance']:.6f}\n")
                    
                    if method_result['model_health_warning']:
                        f.write(f"Model Health Warning: Predictions may not be reliable\n")
                    f.write("\n")
                    
                    if method_result['positive_predictions']:
                        f.write("Positive Detections (>= 50% probability):\n")
                        for pred in method_result['positive_predictions']:
                            f.write(f"  • {pred['disease']}: {pred['probability']:.3f} - {pred['description']}\n")
                    else:
                        f.write("No diseases detected above 50% threshold.\n")
                    
                    f.write(f"\nTop 5 Predictions:\n")
                    for pred in method_result['top_predictions']:
                        f.write(f"  {pred['disease']}: {pred['probability']:.3f} - {pred['description']}\n")
                    
                    f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"Enhanced text report saved to: {txt_path}")


def main():
    """Main function to run the enhanced prediction script"""
    print("=" * 70)
    print("Enhanced Eye Disease Prediction Script")
    print("=" * 70)
    
    try:
        # Initialize predictor
        predictor = EnhancedEyeDiseasePredictor()
        
        # Check if model file exists
        if not os.path.exists("eye_disease_efficientnet_best.h5"):
            print("\nError: Model file 'eye_disease_efficientnet_best.h5' not found!")
            print("Please ensure the trained model file is in the current directory.")
            return
        
        # Check if input folder has images
        image_files = predictor._get_image_files()
        if not image_files:
            print(f"\nInstructions:")
            print(f"1. Place your eye images in the '{predictor.input_folder}' folder")
            print(f"2. Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
            print(f"3. Run this script again")
            return
        
        print(f"\nStarting enhanced prediction process...")
        print(f"Model: EfficientNet for Eye Disease Classification")
        print(f"Disease classes: {len(predictor.disease_classes)} types")
        print(f"Images to process: {len(image_files)}")
        print(f"Model health: {'WARNING' if not predictor.model_is_healthy else 'GOOD'}")
        
        # Choose preprocessing method
        print(f"\nPreprocessing options:")
        print(f"  1. Basic PIL resize and normalize")
        print(f"  2. EfficientNet specific preprocessing")
        print(f"  3. OpenCV with CLAHE enhancement")
        print(f"  0. Try all methods (recommended)")
        
        method_choice = 1  # Default to method 1
        print(f"Using method: {method_choice}")
        
        # Run predictions
        results = predictor.predict_all_images(
            top_k=5,                    # Show top 5 predictions
            threshold=0.5,              # 50% threshold for positive detection
            preprocessing_method=method_choice,  # Preprocessing method
            save_results=True           # Save results to files
        )
        
        if results:
            print(f"\nEnhanced prediction completed successfully!")
            print(f"Processed {len(results)} images")
            print(f"Results saved in '{predictor.output_folder}' folder")
            
            if not predictor.model_is_healthy:
                print(f"\nWARNING: Model health issues detected!")
                print(f"The model may not be making reliable predictions.")
                print(f"Consider:")
                print(f"   • Checking if the model file is corrupted")
                print(f"   • Retraining the model")
                print(f"   • Verifying the model architecture")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check that all files are in place and try again.")


if __name__ == "__main__":
    main()
