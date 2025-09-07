#!/usr/bin/env python3
"""
Firebase Database Setup Script
for Intelligent Attendance Monitoring System
"""

import os
import json
import requests
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("Firebase Database Setup")
    print("Intelligent Attendance Monitoring System")
    print("=" * 60)
    print()

def check_firebase_key():
    """Check if Firebase key file exists"""
    key_file = "./firebase-key.json"
    if os.path.exists(key_file):
        print("✅ Firebase key file found: firebase-key.json")
        return True
    else:
        print("❌ Firebase key file not found: firebase-key.json")
        print("Please download your Firebase service account key and save it as 'firebase-key.json'")
        return False

def create_env_file():
    """Create .env file with Firebase configuration"""
    env_content = """# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=./firebase-key.json
FIREBASE_DATABASE_URL=https://attendance-system-8d32e-default-rtdb.asia-southeast1.firebasedatabase.app/

# Camera Configuration
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# Model Configuration
MODEL_PATH=./data/models/gait_model.h5
CONFIDENCE_THRESHOLD=0.8
FRAME_BUFFER_SIZE=30

# Processing Configuration
FRAME_WIDTH=224
FRAME_HEIGHT=224
BATCH_SIZE=32

# Attendance Configuration
ATTENDANCE_TIMEOUT=300
MAX_STUDENTS_PER_FRAME=10

# Web Dashboard Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/attendance_system.log

# Data Paths
DATA_DIR=./data
TRAINING_DATA_DIR=./data/training_data
TEST_DATA_DIR=./data/test_data
MODELS_DIR=./data/models

# Pose Estimation Configuration
POSE_CONFIDENCE=0.5
POSE_SMOOTHING=0.5

# Feature Extraction Configuration
FEATURE_DIMENSION=512
TEMPORAL_WINDOW=30

# Security Configuration
JWT_SECRET_KEY=your-secret-key-here-change-in-production
SESSION_TIMEOUT=3600

# Performance Configuration
MAX_PROCESSING_TIME=0.1
GPU_MEMORY_FRACTION=0.8
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with Firebase configuration")

def test_firebase_connection():
    """Test Firebase connection"""
    try:
        from firebase_admin import credentials, db
        from firebase_admin import initialize_app
        
        # Initialize Firebase
        cred = credentials.Certificate('./firebase-key.json')
        app = initialize_app(cred, {
            'databaseURL': 'https://attendance-system-8d32e-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
        
        # Test database connection
        ref = db.reference()
        test_data = {'test': 'connection_successful', 'timestamp': '2024-01-01'}
        ref.child('test').set(test_data)
        
        # Clean up test data
        ref.child('test').delete()
        
        print("✅ Firebase connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
        return False

def create_database_structure():
    """Create initial database structure"""
    try:
        from firebase_admin import db
        
        ref = db.reference()
        
        # Create initial structure
        initial_data = {
            'attendance': {},
            'students': {},
            'daily_attendance': {},
            'system_config': {
                'version': '1.0.0',
                'last_updated': '2024-01-01'
            }
        }
        
        ref.set(initial_data)
        print("✅ Database structure created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create database structure: {e}")
        return False

def print_setup_instructions():
    """Print Firebase setup instructions"""
    print("\n" + "=" * 60)
    print("FIREBASE SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Create Firebase Project:")
    print("   - Go to https://console.firebase.google.com/")
    print("   - Click 'Create a project'")
    print("   - Name it 'attendance-system'")
    print("   - Enable Google Analytics (optional)")
    print()
    print("2. Enable Realtime Database:")
    print("   - Go to 'Realtime Database' in Firebase Console")
    print("   - Click 'Create Database'")
    print("   - Choose location (closest to your users)")
    print("   - Start in test mode")
    print()
    print("3. Get Database URL:")
    print("   - Copy the database URL from Realtime Database")
    print("   - Update FIREBASE_DATABASE_URL in .env file")
    print()
    print("4. Generate Service Account Key:")
    print("   - Go to Project Settings (gear icon)")
    print("   - Go to 'Service accounts' tab")
    print("   - Click 'Generate new private key'")
    print("   - Download JSON file")
    print("   - Save as 'firebase-key.json' in project root")
    print()
    print("5. Update .env file:")
    print("   - Update FIREBASE_DATABASE_URL with your database URL")
    print("   - Ensure FIREBASE_CREDENTIALS_PATH points to firebase-key.json")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check if Firebase key exists
    if not check_firebase_key():
        print_setup_instructions()
        return
    
    # Create .env file
    create_env_file()
    
    # Test Firebase connection
    if test_firebase_connection():
        # Create database structure
        create_database_structure()
        
        print("\n" + "=" * 60)
        print("✅ FIREBASE SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Your database is now ready for the attendance system.")
        print("You can now run the system with:")
        print("  python main.py --mode demo")
        print()
        print("Access the web dashboard at: http://localhost:5000")
        print("Login with: admin / admin123")
    else:
        print("\n" + "=" * 60)
        print("❌ FIREBASE SETUP FAILED")
        print("=" * 60)
        print()
        print("Please check your Firebase configuration:")
        print("1. Ensure firebase-key.json is in the project root")
        print("2. Verify the database URL in .env file")
        print("3. Check your internet connection")
        print()
        print_setup_instructions()

if __name__ == "__main__":
    main() 