#!/usr/bin/env python3
"""
Simple test script to verify the Intelligent Attendance System components
"""

import sys
import os
import traceback

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        # Core Python libraries
        import numpy as np
        import cv2
        import json
        import datetime
        print("✓ Core libraries imported")
        
        # Web framework
        from flask import Flask
        print("✓ Flask imported")
        
        # AI/ML libraries
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} imported")
        except ImportError as e:
            print(f"⚠ TensorFlow import failed: {e}")
        
        try:
            import mediapipe as mp
            print(f"✓ MediaPipe {mp.__version__} imported")
        except ImportError as e:
            print(f"⚠ MediaPipe import failed: {e}")
        
        # Project modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            from utils.config import Config
            print("✓ Config module imported")
        except ImportError as e:
            print(f"⚠ Config import failed: {e}")
            return False
            
        try:
            from utils.logger import logger
            print("✓ Logger module imported")
        except ImportError as e:
            print(f"⚠ Logger import failed: {e}")
            return False
            
        try:
            from attendance_system.attendance_tracker import AttendanceTracker
            print("✓ AttendanceTracker imported")
        except ImportError as e:
            print(f"⚠ AttendanceTracker import failed: {e}")
            return False
            
        try:
            from web_dashboard.app import app
            print("✓ Web app imported")
        except ImportError as e:
            print(f"⚠ Web app import failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_directories():
    """Test required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = [
        'data',
        'data/models',
        'data/training_data',
        'data/test_data',
        'logs',
        'web_dashboard/templates',
        'web_dashboard/static'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"⚠ {dir_path} missing - creating...")
            os.makedirs(dir_path, exist_ok=True)
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from utils.config import Config
        print(f"✓ Flask host: {Config.FLASK_HOST}")
        print(f"✓ Flask port: {Config.FLASK_PORT}")
        print(f"✓ Model path: {Config.MODEL_PATH}")
        print(f"✓ Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Intelligent Attendance System - System Test")
    print("=" * 50)
    
    tests = [
        test_directories,
        test_imports,
        test_config
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! System ready to run.")
        return True
    else:
        print("⚠ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
