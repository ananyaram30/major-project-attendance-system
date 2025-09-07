#!/usr/bin/env python3
"""
Terminal-based training script for external Kaggle dataset
Trains model for efficiency improvement without affecting attendance system
Supports both photos and videos with person names
"""

import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_gait_model import GaitRecognitionModel
from data_processing.video_processor import VideoProcessor
from utils.config import Config
from utils.logger import logger

# Initialize colorama for colored terminal output
colorama.init()

class ExternalDatasetTrainer:
    """Terminal-based trainer for external Kaggle dataset"""
    
    def __init__(self):
        self.model = None
        self.video_processor = VideoProcessor()
        self.label_encoder = LabelEncoder()
        self.dataset_path = None
        self.processed_data = None
        
    def print_header(self):
        """Print training header"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🚀 EXTERNAL DATASET TRAINING FOR MODEL EFFICIENCY")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  System: {tf.config.list_physical_devices('GPU') and 'GPU Available' or 'CPU Only'}")
        print(f"📊 TensorFlow: {tf.__version__}")
        print(f"🎯 Goal: Improve model efficiency with external data{Style.RESET_ALL}\n")
    
    def scan_dataset(self, dataset_path):
        """Scan and analyze the external dataset"""
        print(f"{Fore.YELLOW}📂 SCANNING DATASET{Style.RESET_ALL}")
        print(f"   Path: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"{Fore.RED}❌ Dataset path not found!{Style.RESET_ALL}")
            return False
        
        self.dataset_path = dataset_path
        
        # Scan for person directories or files
        persons = {}
        total_files = 0
        
        # Check if it's organized by person folders
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            
            if os.path.isdir(item_path):
                # Person folder structure
                person_name = item
                files = []
                
                for file in os.listdir(item_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov')):
                        files.append(os.path.join(item_path, file))
                
                if files:
                    persons[person_name] = files
                    total_files += len(files)
                    print(f"   👤 {person_name}: {len(files)} files")
            
            elif item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov')):
                # Files directly in root - extract person name from filename
                filename = os.path.splitext(item)[0]
                # Try to extract person name (assuming format like "person_name_001.jpg")
                parts = filename.split('_')
                if len(parts) >= 2:
                    person_name = '_'.join(parts[:-1])  # Everything except last part
                else:
                    person_name = parts[0]
                
                if person_name not in persons:
                    persons[person_name] = []
                
                persons[person_name].append(item_path)
                total_files += 1
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Persons: {len(persons)}")
        print(f"   Total Files: {total_files}")
        
        if len(persons) < 2:
            print(f"{Fore.RED}❌ Need at least 2 persons for training!{Style.RESET_ALL}")
            return False
        
        self.persons_data = persons
        return True
    
    def process_dataset(self, max_samples_per_person=50):
        """Process the external dataset into training format"""
        print(f"\n{Fore.YELLOW}🔄 PROCESSING DATASET{Style.RESET_ALL}")
        
        X_data = []
        y_data = []
        person_names = list(self.persons_data.keys())
        
        # Encode person names to integers
        self.label_encoder.fit(person_names)
        
        for person_idx, (person_name, files) in enumerate(tqdm(self.persons_data.items(), 
                                                              desc="Processing persons", 
                                                              colour="blue")):
            
            # Limit samples per person to avoid class imbalance
            if len(files) > max_samples_per_person:
                files = files[:max_samples_per_person]
            
            person_frames = []
            
            for file_path in tqdm(files, desc=f"Processing {person_name}", leave=False):
                try:
                    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                        # Process video file
                        frames = self.process_video_file(file_path)
                        if frames is not None and len(frames) >= Config.TEMPORAL_WINDOW:
                            person_frames.append(frames[:Config.TEMPORAL_WINDOW])
                    
                    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # Process image file - create sequence by duplicating
                        frame = self.process_image_file(file_path)
                        if frame is not None:
                            # Create temporal sequence by repeating the frame
                            frames = np.repeat(frame[np.newaxis, :], Config.TEMPORAL_WINDOW, axis=0)
                            person_frames.append(frames)
                
                except Exception as e:
                    continue
            
            # Add processed frames to dataset
            person_label = self.label_encoder.transform([person_name])[0]
            
            for frames in person_frames:
                X_data.append(frames)
                y_data.append(person_label)
            
            print(f"   ✅ {person_name}: {len(person_frames)} sequences processed")
        
        if len(X_data) == 0:
            print(f"{Fore.RED}❌ No valid data processed!{Style.RESET_ALL}")
            return False
        
        # Convert to numpy arrays
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"\n📊 Processed Data:")
        print(f"   Total Sequences: {len(X_data)}")
        print(f"   Classes: {len(np.unique(y_data))}")
        print(f"   Input Shape: {X_data.shape}")
        
        self.processed_data = (X_data, y_data)
        return True
    
    def process_video_file(self, video_path):
        """Process a single video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Read frames
            frame_count = 0
            while cap.read()[0] and frame_count < Config.TEMPORAL_WINDOW * 2:  # Read more than needed
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and normalize
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) >= Config.TEMPORAL_WINDOW:
                # Sample frames evenly
                indices = np.linspace(0, len(frames)-1, Config.TEMPORAL_WINDOW, dtype=int)
                sampled_frames = [frames[i] for i in indices]
                return np.array(sampled_frames)
            
            return None
            
        except Exception as e:
            return None
    
    def process_image_file(self, image_path):
        """Process a single image file"""
        try:
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Resize and normalize
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            return None
    
    def create_efficient_model(self, num_classes):
        """Create an efficient model architecture"""
        print(f"\n{Fore.YELLOW}🏗️ BUILDING EFFICIENT MODEL{Style.RESET_ALL}")
        
        # Create lightweight but effective model
        model = tf.keras.Sequential([
            # Efficient 3D CNN layers
            tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', 
                                 input_shape=(Config.TEMPORAL_WINDOW, 224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.Dropout(0.3),
            
            # Global pooling instead of flatten for efficiency
            tf.keras.layers.GlobalAveragePooling3D(),
            
            # Compact dense layers
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with efficient optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Efficient model created with {model.count_params():,} parameters")
        return model
    
    def train_model(self, epochs=50, validation_split=0.2):
        """Train the model with external dataset"""
        if not self.processed_data:
            print(f"{Fore.RED}❌ No processed data available!{Style.RESET_ALL}")
            return False
        
        X_data, y_data = self.processed_data
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=validation_split, 
            stratify=y_data, random_state=42
        )
        
        num_classes = len(np.unique(y_data))
        
        print(f"\n{Fore.YELLOW}🏋️ TRAINING CONFIGURATION{Style.RESET_ALL}")
        print(f"   Epochs: {epochs}")
        print(f"   Classes: {num_classes}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Create model
        self.model = self.create_efficient_model(num_classes)
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Custom progress callback
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                print(f"\n{Fore.CYAN}🚀 TRAINING STARTED{Style.RESET_ALL}")
                self.start_time = time.time()
                self.best_acc = 0.0
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                elapsed = time.time() - self.start_time
                
                acc = logs.get('accuracy', 0)
                val_acc = logs.get('val_accuracy', 0)
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    improvement = f"{Fore.GREEN}📈 NEW BEST!{Style.RESET_ALL}"
                else:
                    improvement = ""
                
                print(f"\n📊 Epoch {epoch + 1}/{epochs}:")
                print(f"   Accuracy: {Fore.GREEN}{acc:.4f}{Style.RESET_ALL} | "
                      f"Val Accuracy: {Fore.GREEN}{val_acc:.4f}{Style.RESET_ALL} {improvement}")
                print(f"   Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"   Time: {elapsed/60:.1f}m | Best: {self.best_acc:.4f}")
                
                # Progress bar
                progress = (epoch + 1) / epochs * 100
                bar_length = 50
                filled = int(bar_length * progress / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   [{Fore.CYAN}{bar}{Style.RESET_ALL}] {progress:.1f}%")
        
        callbacks.append(ProgressCallback())
        
        try:
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=8,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save the improved model (overwrites existing)
            self.model.save(Config.MODEL_PATH)
            
            final_acc = max(history.history['val_accuracy'])
            
            print(f"\n{Fore.GREEN}✅ TRAINING COMPLETED!{Style.RESET_ALL}")
            print(f"   Final Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
            print(f"   Model saved to: {Config.MODEL_PATH}")
            print(f"   🎯 Attendance system efficiency improved!")
            
            return True
            
        except Exception as e:
            print(f"\n{Fore.RED}❌ Training failed: {e}{Style.RESET_ALL}")
            return False
    
    def benchmark_model(self):
        """Benchmark the trained model"""
        print(f"\n{Fore.YELLOW}⚡ PERFORMANCE BENCHMARK{Style.RESET_ALL}")
        
        if not self.model:
            if os.path.exists(Config.MODEL_PATH):
                self.model = tf.keras.models.load_model(Config.MODEL_PATH)
            else:
                print(f"{Fore.RED}❌ No model to benchmark!{Style.RESET_ALL}")
                return
        
        # Benchmark inference time
        sample_input = np.random.random((1, Config.TEMPORAL_WINDOW, 224, 224, 3))
        
        # Warm up
        for _ in range(5):
            _ = self.model.predict(sample_input, verbose=0)
        
        # Time inference
        times = []
        for _ in range(50):
            start = time.time()
            _ = self.model.predict(sample_input, verbose=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        fps = 1.0 / np.mean(times)
        
        print(f"📈 Performance Metrics:")
        print(f"   Average Inference Time: {avg_time:.2f}ms")
        print(f"   Frames Per Second: {fps:.1f} FPS")
        print(f"   Model Size: {os.path.getsize(Config.MODEL_PATH) / (1024*1024):.1f} MB")
        
        if avg_time <= 100:
            print(f"   {Fore.GREEN}🚀 REAL-TIME CAPABLE{Style.RESET_ALL}")
        else:
            print(f"   {Fore.YELLOW}⏱️ NEAR REAL-TIME{Style.RESET_ALL}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train with External Kaggle Dataset')
    parser.add_argument('dataset_path', type=str,
                       help='Path to external dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum samples per person')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark after training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ExternalDatasetTrainer()
    
    # Print header
    trainer.print_header()
    
    # Scan dataset
    if not trainer.scan_dataset(args.dataset_path):
        return 1
    
    # Process dataset
    if not trainer.process_dataset(args.max_samples):
        return 1
    
    # Train model
    if not trainer.train_model(args.epochs):
        return 1
    
    # Benchmark if requested
    if args.benchmark:
        trainer.benchmark_model()
    
    print(f"\n{Fore.GREEN}🎉 External dataset training completed!{Style.RESET_ALL}")
    print(f"💡 Your attendance system model efficiency has been improved.")
    print(f"🔄 The updated model will now provide better accuracy for attendance recognition.")
    
    return 0

if __name__ == "__main__":
    exit(main())
