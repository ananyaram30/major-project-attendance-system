#!/usr/bin/env python3
"""
Terminal-based training script for Attendance System Model
Improves model efficiency and accuracy for better attendance recognition
"""

import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_gait_model import GaitRecognitionModel
from attendance_system.student_manager import StudentManager
from data_processing.video_processor import VideoProcessor
from utils.config import Config
from utils.logger import logger

# Initialize colorama for colored terminal output
colorama.init()

class AttendanceModelTrainer:
    """Terminal-based trainer for attendance recognition model"""
    
    def __init__(self):
        self.model = None
        self.student_manager = StudentManager()
        self.video_processor = VideoProcessor()
        self.training_data = None
        self.validation_data = None
        
    def print_header(self):
        """Print training header"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🎯 ATTENDANCE SYSTEM MODEL TRAINING")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  System: {tf.config.list_physical_devices('GPU') and 'GPU Available' or 'CPU Only'}")
        print(f"📊 TensorFlow: {tf.__version__}")
        print(f"🎯 Goal: Improve attendance recognition accuracy{Style.RESET_ALL}\n")
    
    def check_training_data(self):
        """Check available training data"""
        print(f"{Fore.YELLOW}📊 CHECKING TRAINING DATA{Style.RESET_ALL}")
        
        students = self.student_manager.get_all_students()
        total_students = len(students)
        students_with_data = 0
        total_videos = 0
        
        for student in students:
            student_id = student.get('id', student.get('student_id'))
            if not student_id:
                continue
                
            # Check for training videos
            student_dir = os.path.join(Config.DATA_DIR, 'processed', student_id)
            if os.path.exists(student_dir):
                video_count = len([f for f in os.listdir(student_dir) 
                                 if f.endswith(('.mp4', '.avi', '.mov'))])
                if video_count > 0:
                    students_with_data += 1
                    total_videos += video_count
                    print(f"   ✅ {student.get('name', student_id)}: {video_count} videos")
                else:
                    print(f"   ❌ {student.get('name', student_id)}: No training videos")
            else:
                print(f"   ❌ {student.get('name', student_id)}: No data directory")
        
        print(f"\n📈 Summary:")
        print(f"   Total Students: {total_students}")
        print(f"   Students with Data: {students_with_data}")
        print(f"   Total Training Videos: {total_videos}")
        
        if students_with_data < 2:
            print(f"{Fore.RED}❌ Need at least 2 students with training data!{Style.RESET_ALL}")
            return False
        
        return True
    
    def prepare_training_data(self):
        """Prepare training data from student videos"""
        print(f"\n{Fore.YELLOW}🔄 PREPARING TRAINING DATA{Style.RESET_ALL}")
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        class_names = []
        
        students = self.student_manager.get_all_students()
        class_id = 0
        
        for student in tqdm(students, desc="Processing students", colour="blue"):
            student_id = student.get('id', student.get('student_id'))
            student_name = student.get('name', student_id)
            
            if not student_id:
                continue
            
            student_dir = os.path.join(Config.DATA_DIR, 'processed', student_id)
            if not os.path.exists(student_dir):
                continue
            
            # Load processed frames for this student
            frame_files = [f for f in os.listdir(student_dir) 
                          if f.endswith('.npy')]
            
            if len(frame_files) < 10:  # Need minimum frames
                continue
            
            class_names.append(student_name)
            student_frames = []
            
            for frame_file in frame_files:
                frame_path = os.path.join(student_dir, frame_file)
                try:
                    frames = np.load(frame_path)
                    if frames.shape[0] >= Config.TEMPORAL_WINDOW:
                        student_frames.append(frames[:Config.TEMPORAL_WINDOW])
                except Exception as e:
                    continue
            
            if len(student_frames) == 0:
                continue
            
            # Split into train/validation (80/20)
            split_idx = int(0.8 * len(student_frames))
            
            # Training data
            for frames in student_frames[:split_idx]:
                X_train.append(frames)
                y_train.append(class_id)
            
            # Validation data
            for frames in student_frames[split_idx:]:
                X_val.append(frames)
                y_val.append(class_id)
            
            class_id += 1
        
        if len(X_train) == 0:
            print(f"{Fore.RED}❌ No training data found!{Style.RESET_ALL}")
            return False
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val) if X_val else X_train[:5]  # Use some training data if no validation
        y_val = np.array(y_val) if y_val else y_train[:5]
        
        self.training_data = (X_train, y_train)
        self.validation_data = (X_val, y_val)
        
        print(f"✅ Training data prepared:")
        print(f"   Classes: {len(class_names)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Input shape: {X_train.shape[1:]}")
        
        return True
    
    def create_improved_model(self, num_classes):
        """Create an improved model architecture"""
        print(f"\n{Fore.YELLOW}🏗️ BUILDING IMPROVED MODEL{Style.RESET_ALL}")
        
        # Create enhanced model with better architecture
        model = tf.keras.Sequential([
            # 3D CNN layers for temporal features
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', 
                                 input_shape=(Config.TEMPORAL_WINDOW, 224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.Dropout(0.3),
            
            # Flatten and add LSTM for sequence modeling
            tf.keras.layers.Reshape((-1, 128)),
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3),
            tf.keras.layers.LSTM(128, dropout=0.3),
            
            # Dense layers with regularization
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def create_training_callbacks(self):
        """Create training callbacks for better performance"""
        callbacks = []
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            Config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Custom progress callback
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.start_time = None
                self.best_acc = 0.0
                
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                print(f"\n{Fore.CYAN}🚀 TRAINING STARTED{Style.RESET_ALL}")
                
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
                
                print(f"\n📊 Epoch {epoch + 1} Results:")
                print(f"   Accuracy: {Fore.GREEN}{acc:.4f}{Style.RESET_ALL} | "
                      f"Val Accuracy: {Fore.GREEN}{val_acc:.4f}{Style.RESET_ALL} {improvement}")
                print(f"   Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"   Time: {elapsed/60:.1f}m | Best Val Acc: {self.best_acc:.4f}")
                
                # Progress bar
                if hasattr(self, 'total_epochs'):
                    progress = (epoch + 1) / self.total_epochs * 100
                    bar_length = 50
                    filled = int(bar_length * progress / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"   [{Fore.CYAN}{bar}{Style.RESET_ALL}] {progress:.1f}%")
        
        progress_cb = ProgressCallback()
        callbacks.append(progress_cb)
        
        return callbacks
    
    def train_model(self, epochs=100):
        """Train the model with progress display"""
        if not self.training_data or not self.validation_data:
            print(f"{Fore.RED}❌ No training data available!{Style.RESET_ALL}")
            return False
        
        X_train, y_train = self.training_data
        X_val, y_val = self.validation_data
        
        num_classes = len(np.unique(y_train))
        
        print(f"\n{Fore.YELLOW}🏋️ TRAINING CONFIGURATION{Style.RESET_ALL}")
        print(f"   Epochs: {epochs}")
        print(f"   Classes: {num_classes}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Create model
        self.model = self.create_improved_model(num_classes)
        
        # Create callbacks
        callbacks = self.create_training_callbacks()
        callbacks[3].total_epochs = epochs  # Set total epochs for progress callback
        
        try:
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=8,
                callbacks=callbacks,
                verbose=0  # We handle progress in callback
            )
            
            # Get final results
            final_acc = max(history.history['val_accuracy'])
            
            print(f"\n{Fore.GREEN}✅ TRAINING COMPLETED!{Style.RESET_ALL}")
            print(f"   Final Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
            print(f"   Model saved to: {Config.MODEL_PATH}")
            
            return True
            
        except Exception as e:
            print(f"\n{Fore.RED}❌ Training failed: {e}{Style.RESET_ALL}")
            return False
    
    def test_model(self):
        """Test the trained model"""
        print(f"\n{Fore.YELLOW}🧪 TESTING MODEL{Style.RESET_ALL}")
        
        if not self.model:
            # Try to load existing model
            if os.path.exists(Config.MODEL_PATH):
                self.model = tf.keras.models.load_model(Config.MODEL_PATH)
                print("✅ Loaded existing model")
            else:
                print(f"{Fore.RED}❌ No trained model found!{Style.RESET_ALL}")
                return False
        
        if not self.validation_data:
            print(f"{Fore.RED}❌ No test data available!{Style.RESET_ALL}")
            return False
        
        X_val, y_val = self.validation_data
        
        # Evaluate model
        results = self.model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = results[0], results[1]
        
        # Get predictions for detailed analysis
        predictions = self.model.predict(X_val, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate per-class accuracy
        unique_classes = np.unique(y_val)
        class_accuracies = []
        
        for class_id in unique_classes:
            class_mask = y_val == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predicted_classes[class_mask] == y_val[class_mask])
                class_accuracies.append(class_acc)
        
        avg_class_acc = np.mean(class_accuracies) if class_accuracies else 0
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   Overall Accuracy: {Fore.GREEN}{test_acc:.4f} ({test_acc*100:.2f}%){Style.RESET_ALL}")
        print(f"   Average Class Accuracy: {avg_class_acc:.4f} ({avg_class_acc*100:.2f}%)")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Classes Tested: {len(unique_classes)}")
        
        # Efficiency assessment
        if test_acc >= 0.90:
            print(f"   {Fore.GREEN}🎯 EXCELLENT EFFICIENCY (≥90%){Style.RESET_ALL}")
        elif test_acc >= 0.80:
            print(f"   {Fore.YELLOW}⚡ GOOD EFFICIENCY (≥80%){Style.RESET_ALL}")
        elif test_acc >= 0.70:
            print(f"   {Fore.ORANGE}⚠️ MODERATE EFFICIENCY (≥70%){Style.RESET_ALL}")
        else:
            print(f"   {Fore.RED}❌ LOW EFFICIENCY (<70%){Style.RESET_ALL}")
        
        return test_acc
    
    def benchmark_performance(self):
        """Benchmark model performance for attendance system"""
        print(f"\n{Fore.YELLOW}⚡ PERFORMANCE BENCHMARK{Style.RESET_ALL}")
        
        if not self.model:
            if os.path.exists(Config.MODEL_PATH):
                self.model = tf.keras.models.load_model(Config.MODEL_PATH)
            else:
                print(f"{Fore.RED}❌ No model to benchmark!{Style.RESET_ALL}")
                return
        
        # Create sample input for timing
        sample_input = np.random.random((1, Config.TEMPORAL_WINDOW, 224, 224, 3))
        
        # Warm up
        for _ in range(5):
            _ = self.model.predict(sample_input, verbose=0)
        
        # Benchmark inference time
        times = []
        for _ in range(100):
            start = time.time()
            _ = self.model.predict(sample_input, verbose=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        fps = 1.0 / np.mean(times)
        
        print(f"📈 Performance Metrics:")
        print(f"   Average Inference Time: {avg_time:.2f}ms")
        print(f"   Frames Per Second: {fps:.1f} FPS")
        print(f"   Model Size: {os.path.getsize(Config.MODEL_PATH) / (1024*1024):.1f} MB")
        
        if avg_time <= 100:  # 100ms threshold
            print(f"   {Fore.GREEN}🚀 REAL-TIME CAPABLE{Style.RESET_ALL}")
        else:
            print(f"   {Fore.YELLOW}⏱️ NEAR REAL-TIME{Style.RESET_ALL}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Attendance Recognition Model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AttendanceModelTrainer()
    
    # Print header
    trainer.print_header()
    
    if args.test_only:
        # Only test existing model
        accuracy = trainer.test_model()
        if args.benchmark:
            trainer.benchmark_performance()
        return 0 if accuracy and accuracy >= 0.7 else 1
    
    # Check training data
    if not trainer.check_training_data():
        print(f"\n{Fore.RED}❌ Insufficient training data. Please upload student videos first.{Style.RESET_ALL}")
        return 1
    
    # Prepare training data
    if not trainer.prepare_training_data():
        return 1
    
    # Train model
    if not trainer.train_model(args.epochs):
        return 1
    
    # Test model
    accuracy = trainer.test_model()
    
    # Benchmark if requested
    if args.benchmark:
        trainer.benchmark_performance()
    
    print(f"\n{Fore.GREEN}🎉 Training completed successfully!{Style.RESET_ALL}")
    print(f"💡 The improved model will enhance attendance system accuracy.")
    
    return 0 if accuracy and accuracy >= 0.7 else 1

if __name__ == "__main__":
    exit(main())
