#!/usr/bin/env python3
"""
Training script for the Intelligent Attendance System
CNN-based Gait Recognition Model
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.cnn_gait_model import GaitRecognitionModel, ModelEvaluator
from utils.config import Config
from utils.logger import logger

def load_sample_data(data_path: str, num_classes: int, num_samples: int = 100):
    """Load sample training data (placeholder for real data loading)"""
    logger.info(f"Loading sample data from {data_path}")
    
    # Generate synthetic data for demonstration
    # In a real implementation, you would load actual video data
    sample_data = np.random.random((num_samples, Config.TEMPORAL_WINDOW, Config.FRAME_WIDTH, Config.FRAME_HEIGHT, 3))
    sample_labels = np.random.randint(0, num_classes, (num_samples,))
    
    logger.info(f"Generated {num_samples} samples with {num_classes} classes")
    return sample_data, sample_labels

def train_model(args):
    """Train the gait recognition model"""
    try:
        logger.info("Starting model training...")
        
        # Initialize model
        model = GaitRecognitionModel(num_classes=args.num_classes)
        
        # Build model
        logger.info(f"Building {args.model_type} model...")
        model.build_model(model_type=args.model_type)
        
        # Load training data
        train_data, train_labels = load_sample_data(args.data_path, args.num_classes, args.num_samples)
        
        # Convert labels to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        train_labels_onehot = to_categorical(train_labels, args.num_classes)
        
        # Split data into train and validation
        split_idx = int(0.8 * len(train_data))
        train_x, train_y = train_data[:split_idx], train_labels_onehot[:split_idx]
        val_x, val_y = train_data[split_idx:], train_labels_onehot[split_idx:]
        
        logger.info(f"Training data: {len(train_x)} samples")
        logger.info(f"Validation data: {len(val_y)} samples")
        
        # Train model
        logger.info("Starting training...")
        history = model.train(
            train_data=(train_x, train_y),
            val_data=(val_x, val_y),
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save model
        model.save_model()
        logger.info("Model saved successfully")
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = ModelEvaluator.evaluate_model(model.model, (val_x, val_y))
        
        logger.info(f"Validation Accuracy: {evaluation_results['accuracy']:.3f}")
        logger.info(f"Validation Loss: {evaluation_results['loss']:.3f}")
        
        # Plot training history
        ModelEvaluator.plot_training_history(history)
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Gait Recognition Model")
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/training_data',
        help='Path to training data'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['hybrid', '3d_cnn', 'lstm'],
        default='hybrid',
        help='Model architecture type'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='Number of classes/students'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of training samples to generate'
    )
    
    args = parser.parse_args()
    
    logger.info("=== Gait Recognition Model Training ===")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Number of Classes: {args.num_classes}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Data Path: {args.data_path}")
    
    # Create necessary directories
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    
    # Train model
    success = train_model(args)
    
    if success:
        logger.info("Training completed successfully!")
        return 0
    else:
        logger.error("Training failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 