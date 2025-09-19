"""
EfficientNet Eye Disease Classification Training Script - FIXED VERSION
====================================================================
A comprehensive, easy-to-understand script for training an EfficientNet model 
on the RFMiD eye disease dataset with robust error handling.

This version includes fixes for common TensorFlow issues.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sklearn for metrics
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ====================================================================
# CONFIGURATION PARAMETERS
# ====================================================================

class Config:
    """Configuration class containing all hyperparameters and paths"""
    
    # Dataset paths
    TRAIN_DIR = "database/Training_Set/Training_Set/Training"
    TRAIN_CSV = "database/Training_Set/Training_Set/RFMiD_Training_Labels.csv"
    VAL_DIR = "database/Evaluation_Set/Evaluation_Set/Validation"
    VAL_CSV = "database/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"
    TEST_DIR = "database/Test_Set/Test_Set/Test"
    TEST_CSV = "database/Test_Set/Test_Set/RFMiD_Testing_Labels.csv"
    
    # Model parameters
    IMG_SIZE = 224  # EfficientNet-B0 input size
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Model saving
    MODEL_NAME = "eye_disease_efficientnet_best.h5"
    
    # Disease classes (excluding ID and Disease_Risk columns)
    DISEASE_CLASSES = [
        'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS',
        'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
        'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM',
        'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA',
        'VS', 'BRAO', 'PLQ', 'HPED', 'CL'
    ]
    
    NUM_CLASSES = len(DISEASE_CLASSES)  # 45 disease classes

# ====================================================================
# DATA LOADING AND PREPROCESSING
# ====================================================================

class EyeDiseaseDataset:
    """Handle dataset loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self):
        """Load CSV files and prepare dataframes"""
        print("Loading dataset...")
        
        # Load training data
        self.train_df = pd.read_csv(self.config.TRAIN_CSV)
        print(f"Training samples: {len(self.train_df)}")
        
        # Load validation data
        self.val_df = pd.read_csv(self.config.VAL_CSV)
        print(f"Validation samples: {len(self.val_df)}")
        
        # Load test data
        self.test_df = pd.read_csv(self.config.TEST_CSV)
        print(f"Test samples: {len(self.test_df)}")
        
        # Display basic statistics
        self._display_statistics()
        
    def _display_statistics(self):
        """Display dataset statistics"""
        print("\nDataset Statistics:")
        print(f"Total training images: {len(self.train_df)}")
        print(f"Total validation images: {len(self.val_df)}")
        print(f"Total test images: {len(self.test_df)}")
        print(f"Number of disease classes: {self.config.NUM_CLASSES}")
        
        # Count positive cases for each disease in training set
        disease_counts = self.train_df[self.config.DISEASE_CLASSES].sum()
        print("\nDisease distribution in training set (top 10):")
        print(disease_counts.sort_values(ascending=False).head(10))
        
    def create_generators(self):
        """Create data generators for training and validation"""
        print("\nCreating data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Prepare dataframes with proper file paths
        train_df_gen = self._prepare_dataframe(self.train_df, self.config.TRAIN_DIR)
        val_df_gen = self._prepare_dataframe(self.val_df, self.config.VAL_DIR)
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df_gen,
            x_col='filename',
            y_col=self.config.DISEASE_CLASSES,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='raw',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df_gen,
            x_col='filename',
            y_col=self.config.DISEASE_CLASSES,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='raw',
            shuffle=False
        )
        
        print(f"Training generator: {len(train_generator)} batches")
        print(f"Validation generator: {len(val_generator)} batches")
        
        return train_generator, val_generator
    
    def _prepare_dataframe(self, df, img_dir):
        """Prepare dataframe with full file paths"""
        df_copy = df.copy()
        df_copy['filename'] = df_copy['ID'].apply(lambda x: os.path.join(img_dir, f"{x}.png"))
        return df_copy

# ====================================================================
# MODEL BUILDING - FIXED VERSION
# ====================================================================

class EfficientNetModel:
    """EfficientNet model builder for multi-label classification - FIXED VERSION"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def build_model(self, use_pretrained=True):
        """Build EfficientNet-B0 model for multi-label classification with error handling"""
        print("\nBuilding EfficientNet-B0 model...")
        
        try:
            if use_pretrained:
                # Try to load pre-trained EfficientNet-B0
                print("Attempting to load pre-trained EfficientNet-B0 weights...")
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3)
                )
                print("Successfully loaded pre-trained weights!")
            else:
                raise Exception("Using random weights as requested")
                
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
            print("Building model with random weights (training will take longer)...")
            
            # Build EfficientNet without pre-trained weights
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3)
            )
        
        # Freeze the base model initially (if using pretrained weights)
        if use_pretrained:
            base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='sigmoid')(x)  # Sigmoid for multi-label
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',  # Binary crossentropy for multi-label
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def unfreeze_model(self):
        """Unfreeze the base model for fine-tuning"""
        print("\nUnfreezing base model for fine-tuning...")
        
        base_model = self.model.layers[1]  # EfficientNet base model
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE/10),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning from layer {fine_tune_at} onwards")

# ====================================================================
# ALTERNATIVE SIMPLE MODEL (FALLBACK)
# ====================================================================

class SimpleModel:
    """Simple CNN model as fallback if EfficientNet fails"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def build_model(self):
        """Build a simple CNN model as fallback"""
        print("\nBuilding Simple CNN model (fallback option)...")
        
        model = keras.Sequential([
            keras.Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3)),
            
            # First block
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            # Second block  
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            # Third block
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            # Fourth block
            layers.Conv2D(256, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            
            # Classification head
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.config.NUM_CLASSES, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Simple CNN model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model

# ====================================================================
# TRAINING
# ====================================================================

class ModelTrainer:
    """Handle model training process"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = None
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            ModelCheckpoint(
                filepath=self.config.MODEL_NAME,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.5,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                verbose=1,
                restore_best_weights=True
            )
        ]
        return callbacks
    
    def train(self, train_generator, val_generator, fine_tune=True):
        """Train the model"""
        print("\nStarting training...")
        
        callbacks = self.setup_callbacks()
        
        # Initial training with frozen base
        print("Phase 1: Training with frozen base model")
        history1 = self.model.fit(
            train_generator,
            epochs=self.config.EPOCHS // 2,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        if fine_tune:
            # Unfreeze and fine-tune (only for EfficientNet)
            if hasattr(self.model, 'layers') and len(self.model.layers) > 1:
                efficient_model = EfficientNetModel(self.config)
                efficient_model.model = self.model
                efficient_model.unfreeze_model()
                
                print("\nPhase 2: Fine-tuning unfrozen model")
                history2 = self.model.fit(
                    train_generator,
                    epochs=self.config.EPOCHS // 2,
                    validation_data=val_generator,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Combine histories
                self.history = self._combine_histories(history1, history2)
            else:
                print("\nSkipping fine-tuning for simple model")
                self.history = history1
        else:
            self.history = history1
            
        print("Training completed!")
        return self.history
    
    def _combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined_history = {}
        for key in hist1.history.keys():
            combined_history[key] = hist1.history[key] + hist2.history[key]
        return type('History', (), {'history': combined_history})()

# ====================================================================
# EVALUATION
# ====================================================================

class ModelEvaluator:
    """Handle model evaluation and metrics"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def evaluate_model(self, val_generator):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Get predictions
        predictions = self.model.predict(val_generator, verbose=1)
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Get true labels
        val_generator.reset()
        y_true = []
        for i in range(len(val_generator)):
            _, batch_labels = val_generator[i]
            y_true.extend(batch_labels)
        y_true = np.array(y_true)
        
        # Calculate metrics
        self._calculate_metrics(y_true, binary_predictions, predictions)
        
        # Plot results
        self._plot_training_history()
        
        return predictions, binary_predictions
    
    def _calculate_metrics(self, y_true, y_pred_binary, y_pred_proba):
        """Calculate and display various metrics"""
        print("\nModel Performance Metrics:")
        
        # Hamming Loss (lower is better)
        hamming = hamming_loss(y_true, y_pred_binary)
        print(f"Hamming Loss: {hamming:.4f}")
        
        # Per-class metrics
        print("\nPer-Disease Classification Report:")
        report = classification_report(
            y_true, y_pred_binary,
            target_names=self.config.DISEASE_CLASSES,
            output_dict=True,
            zero_division=0
        )
        
        # Display summary metrics
        print(f"Macro Average Precision: {report['macro avg']['precision']:.3f}")
        print(f"Macro Average Recall: {report['macro avg']['recall']:.3f}")
        print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.3f}")
        
        # Show top performing diseases
        disease_f1 = [(disease, report[disease]['f1-score']) 
                     for disease in self.config.DISEASE_CLASSES 
                     if disease in report]
        disease_f1.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Best Performing Diseases (F1-Score):")
        for disease, f1 in disease_f1[:10]:
            print(f"{disease}: {f1:.3f}")
    
    def _plot_training_history(self):
        """Plot training history"""
        if not hasattr(self, 'history') or self.history is None:
            return
            
        print("\nTraining history plots saved as 'training_history.png'")
        
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot training & validation accuracy
            plt.subplot(1, 3, 1)
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot training & validation loss
            plt.subplot(1, 3, 2)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot learning rate if available
            if 'lr' in self.history.history:
                plt.subplot(1, 3, 3)
                plt.plot(self.history.history['lr'])
                plt.title('Learning Rate')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Warning: Could not save training plots: {e}")

# ====================================================================
# MAIN TRAINING PIPELINE - FIXED VERSION
# ====================================================================

def main():
    """Main training pipeline with robust error handling"""
    print("EfficientNet Eye Disease Classification Training - FIXED VERSION")
    print("=" * 70)
    
    # Initialize configuration
    config = Config()
    
    # Check if GPU is available
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Load dataset
    dataset = EyeDiseaseDataset(config)
    dataset.load_data()
    
    # Create data generators
    train_gen, val_gen = dataset.create_generators()
    
    # Build model with error handling
    model = None
    model_type = "unknown"
    
    try:
        # Try EfficientNet first
        print("\n=== Attempting to build EfficientNet model ===")
        efficient_model = EfficientNetModel(config)
        model = efficient_model.build_model(use_pretrained=True)
        model_type = "EfficientNet (pretrained)"
        
    except Exception as e1:
        print(f"EfficientNet with pretrained weights failed: {e1}")
        
        try:
            # Try EfficientNet without pretrained weights
            print("\n=== Attempting EfficientNet without pretrained weights ===")
            efficient_model = EfficientNetModel(config)
            model = efficient_model.build_model(use_pretrained=False)
            model_type = "EfficientNet (random weights)"
            
        except Exception as e2:
            print(f"EfficientNet without pretrained weights failed: {e2}")
            
            try:
                # Fallback to simple CNN
                print("\n=== Using Simple CNN as fallback ===")
                simple_model = SimpleModel(config)
                model = simple_model.build_model()
                model_type = "Simple CNN"
                
            except Exception as e3:
                print(f"Even simple model failed: {e3}")
                print("CRITICAL ERROR: Cannot build any model!")
                return None, None, None
    
    print(f"\nSUCCESS: Using {model_type}")
    
    # Display model summary
    print("\nModel Summary:")
    try:
        model.summary()
    except:
        print("Could not display model summary")
    
    # Setup trainer
    trainer = ModelTrainer(model, config)
    
    # Train model
    fine_tune = model_type.startswith("EfficientNet")
    history = trainer.train(train_gen, val_gen, fine_tune=fine_tune)
    
    # Evaluate model
    evaluator = ModelEvaluator(model, config)
    evaluator.history = history  # Pass history for plotting
    predictions, binary_predictions = evaluator.evaluate_model(val_gen)
    
    # Save final model
    print(f"\nSaving final model as '{config.MODEL_NAME}'")
    try:
        model.save(config.MODEL_NAME)
        print(f"Model saved successfully!")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
    
    print(f"\nTraining completed successfully using {model_type}!")
    print(f"Best model saved as: {config.MODEL_NAME}")
    print("Training history plot saved as: training_history.png")
    
    return model, history, predictions

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def predict_single_image(model_path, image_path, config):
    """Predict diseases for a single image"""
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)[0]
    
    # Get top diseases
    disease_probs = list(zip(config.DISEASE_CLASSES, predictions))
    disease_probs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nPredictions for {image_path}:")
    print("Top 5 predicted diseases:")
    for disease, prob in disease_probs[:5]:
        if prob > 0.1:  # Show only significant predictions
            print(f"  {disease}: {prob:.3f}")

def load_and_test_model(model_path="eye_disease_efficientnet_best.h5"):
    """Load saved model and test on validation set"""
    config = Config()
    
    # Load model
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load validation data
    dataset = EyeDiseaseDataset(config)
    dataset.load_data()
    _, val_gen = dataset.create_generators()
    
    # Evaluate
    evaluator = ModelEvaluator(model, config)
    predictions, binary_predictions = evaluator.evaluate_model(val_gen)
    
    return model, predictions

# ====================================================================
# RUN TRAINING
# ====================================================================

if __name__ == "__main__":
    # Run main training pipeline
    model, history, predictions = main()
    
    if model is not None:
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("TRAINING FAILED - Please check the error messages above")
        print("="*50)
