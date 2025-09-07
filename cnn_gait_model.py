import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
import numpy as np
from typing import Tuple, Optional, List, Dict
from utils.config import Config
from utils.logger import logger
import os
import json

class GaitRecognitionModel:
    """CNN-based gait recognition model for attendance monitoring"""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.feature_extractor = None
        self.classifier = None
        
    def build_feature_extractor(self) -> tf.keras.Model:
        """Build CNN feature extractor using ResNet50V2"""
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers for gait feature extraction
        x = base_model.output
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        
        return models.Model(inputs=base_model.input, outputs=x)
    
    def build_temporal_model(self, feature_dim: int, sequence_length: int = 30) -> tf.keras.Model:
        """Build LSTM model for temporal feature processing"""
        model = models.Sequential([
            layers.LSTM(256, return_sequences=True, input_shape=(sequence_length, feature_dim)),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def build_3d_cnn_model(self) -> tf.keras.Model:
        """Build 3D CNN model for spatio-temporal feature extraction"""
        model = models.Sequential([
            # 3D Convolutional layers
            layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(30, 224, 224, 3)),
            layers.MaxPooling3D((1, 2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv3D(64, (3, 3, 3), activation='relu'),
            layers.MaxPooling3D((1, 2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv3D(128, (3, 3, 3), activation='relu'),
            layers.MaxPooling3D((1, 2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv3D(256, (3, 3, 3), activation='relu'),
            layers.MaxPooling3D((1, 2, 2)),
            layers.BatchNormalization(),
            
            # Global average pooling
            layers.GlobalAveragePooling3D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def build_hybrid_model(self) -> tf.keras.Model:
        """Build hybrid CNN-LSTM model"""
        # Feature extractor
        feature_extractor = self.build_feature_extractor()
        
        # Temporal processing
        temporal_input = layers.Input(shape=(Config.TEMPORAL_WINDOW, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        
        # Process each frame through feature extractor
        features = []
        for i in range(Config.TEMPORAL_WINDOW):
            frame_features = feature_extractor(temporal_input[:, i, :, :, :])
            features.append(frame_features)
        
        # Stack features for LSTM
        stacked_features = layers.Concatenate()(features)
        reshaped_features = layers.Reshape((Config.TEMPORAL_WINDOW, -1))(stacked_features)
        
        # LSTM layers
        lstm_out = layers.LSTM(256, return_sequences=True)(reshaped_features)
        lstm_out = layers.Dropout(0.3)(lstm_out)
        lstm_out = layers.LSTM(128, return_sequences=False)(reshaped_features)
        lstm_out = layers.Dropout(0.3)(lstm_out)
        
        # Classification layers
        dense_out = layers.Dense(256, activation='relu')(lstm_out)
        dense_out = layers.Dropout(0.5)(dense_out)
        dense_out = layers.Dense(128, activation='relu')(dense_out)
        dense_out = layers.Dropout(0.3)(dense_out)
        output = layers.Dense(self.num_classes, activation='softmax')(dense_out)
        
        model = models.Model(inputs=temporal_input, outputs=output)
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_model(self, model_type: str = 'hybrid') -> tf.keras.Model:
        """Build the complete gait recognition model"""
        if model_type == '3d_cnn':
            self.model = self.build_3d_cnn_model()
        elif model_type == 'lstm':
            feature_extractor = self.build_feature_extractor()
            self.feature_extractor = feature_extractor
            self.classifier = self.build_temporal_model(Config.FEATURE_DIMENSION)
        else:  # hybrid
            self.model = self.build_hybrid_model()
        
        if self.model:
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
            )
        
        logger.info(f"Built {model_type} gait recognition model with {self.num_classes} classes")
        return self.model
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              val_data: Tuple[np.ndarray, np.ndarray],
              epochs: int = 100,
              batch_size: int = 32,
              callbacks: list = None) -> tf.keras.callbacks.History:
        """Train the model"""
        if not self.model:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
                tf.keras.callbacks.ModelCheckpoint(
                    Config.MODEL_PATH,
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
        
        history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
    
    def predict(self, input_data: np.ndarray) -> Tuple[int, float]:
        """Predict student identity from input data"""
        try:
            if self.model is None:
                logger.warning("Model not loaded")
                return 0, 0.0
            
            # Preprocess input
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)
            
            # Normalize
            input_data = input_data.astype(np.float32) / 255.0
            
            # Predict
            predictions = self.model.predict(input_data)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0, 0.0
    
    def train_model(self, X_train: List[np.ndarray], y_train: List[str]) -> bool:
        """Train the model with video frames and student IDs"""
        try:
            if not X_train or not y_train:
                logger.error("No training data provided")
                return False
            
            logger.info(f"Starting model training with {len(X_train)} frames for {len(set(y_train))} students")
            
            # Convert student IDs to class indices
            unique_students = list(set(y_train))
            student_to_class = {student: idx for idx, student in enumerate(unique_students)}
            y_encoded = [student_to_class[student_id] for student_id in y_train]
            
            # Convert to numpy arrays
            X = np.array(X_train)
            y = np.array(y_encoded)
            
            logger.info(f"Input data shape: {X.shape}")
            logger.info(f"Labels shape: {y.shape}")
            
            # Check if we have enough data
            if len(X) < 10:
                logger.error(f"Not enough training data: {len(X)} samples (need at least 10)")
                return False
            
            # Normalize input data
            X = X.astype(np.float32) / 255.0
            
            # Convert labels to categorical
            from tensorflow.keras.utils import to_categorical
            y_categorical = to_categorical(y, num_classes=len(unique_students))
            
            # Build model if not exists
            if self.model is None:
                self.num_classes = len(unique_students)
                logger.info(f"Building hybrid model for {self.num_classes} classes")
                self.model = self.build_hybrid_model()
            
            if self.model is None:
                logger.error("Failed to build model")
                return False
            
            logger.info("Model built successfully")
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Starting model training...")
            
            # Train model with fewer epochs for testing
            history = self.model.fit(
                X, y_categorical,
                epochs=5,  # Reduced from 50 for testing
                batch_size=min(32, len(X)),
                validation_split=0.2 if len(X) > 10 else 0.0,
                verbose=1
            )
            
            # Save model
            self.save_model()
            
            # Save student mapping
            self._save_student_mapping(student_to_class)
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _save_student_mapping(self, student_to_class: Dict[str, int]):
        """Save student ID to class index mapping"""
        try:
            mapping_file = "./data/student_class_mapping.json"
            os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
            
            with open(mapping_file, 'w') as f:
                json.dump(student_to_class, f, indent=2)
            
            logger.info(f"Student mapping saved to {mapping_file}")
            
        except Exception as e:
            logger.error(f"Error saving student mapping: {e}")
    
    def load_student_mapping(self) -> Dict[str, int]:
        """Load student ID to class index mapping"""
        try:
            mapping_file = "./data/student_class_mapping.json"
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    return json.load(f)
            return {}
            
        except Exception as e:
            logger.error(f"Error loading student mapping: {e}")
            return {}
    
    def predict_single_frame(self, frame: np.ndarray) -> Tuple[int, float]:
        """Predict identity for a single frame"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        predicted_class, confidence = self.predict(processed_frame)
        
        return predicted_class[0], confidence[0]
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess single frame for prediction"""
        # Resize to model input size
        resized = tf.image.resize(frame, (self.input_shape[0], self.input_shape[1]))
        # Normalize
        normalized = resized / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def preprocess_sequence(self, frames: list) -> np.ndarray:
        """Preprocess sequence of frames for prediction"""
        processed_frames = []
        for frame in frames:
            processed = self.preprocess_frame(frame)
            processed_frames.append(processed)
        
        # Stack frames along time dimension
        sequence = np.concatenate(processed_frames, axis=0)
        return np.expand_dims(sequence, axis=0)
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if filepath is None:
            filepath = Config.MODEL_PATH
        
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.error("No model to save")
    
    def load_model(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = Config.MODEL_PATH
        
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model:
            summary = []
            self.model.summary(print_fn=lambda x: summary.append(x))
            return '\n'.join(summary)
        return "Model not built yet"

class ModelEvaluator:
    """Helper class for model evaluation"""
    
    @staticmethod
    def evaluate_model(model: tf.keras.Model, test_data: Tuple[np.ndarray, np.ndarray]) -> dict:
        """Evaluate model performance"""
        loss, accuracy, top_k_accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
        
        # Get predictions
        predictions = model.predict(test_data[0], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_data[1], axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(true_classes, predicted_classes, output_dict=True)
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'top_k_accuracy': top_k_accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    @staticmethod
    def plot_training_history(history: tf.keras.callbacks.History):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('./data/models/training_history.png')
        plt.close()
        
        logger.info("Training history plot saved") 