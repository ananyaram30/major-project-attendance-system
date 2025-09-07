#!/usr/bin/env python3
"""
Setup script for Intelligent Attendance Monitoring System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("Intelligent Attendance Monitoring System Setup")
    print("Using CNN-based Gait Recognition")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/training_data",
        "data/test_data", 
        "data/models",
        "data/attendance_images",
        "data/reports",
        "data/exports",
        "data/backups",
        "logs",
        "web_dashboard/static",
        "web_dashboard/templates"
    ]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    print()

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    print()
    return True

def create_env_file():
    """Create .env file from example"""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✅ Created .env file")
        print("⚠️  Please edit .env file with your configuration")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No env_example.txt found, creating basic .env file...")
        with open(env_file, "w") as f:
            f.write("# Basic configuration\n")
            f.write("FLASK_HOST=0.0.0.0\n")
            f.write("FLASK_PORT=5000\n")
            f.write("FLASK_DEBUG=True\n")
        print("✅ Created basic .env file")
    print()

def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    # Create sample students data
    sample_students = {
        "STU001": {
            "student_id": "STU001",
            "name": "John Doe",
            "email": "john.doe@university.edu",
            "phone": "+1234567890",
            "age": 20,
            "gender": "Male",
            "department": "Computer Science",
            "year": "2024"
        },
        "STU002": {
            "student_id": "STU002", 
            "name": "Jane Smith",
            "email": "jane.smith@university.edu",
            "phone": "+1234567891",
            "age": 19,
            "gender": "Female",
            "department": "Engineering",
            "year": "2024"
        }
    }
    
    import json
    with open("data/students.json", "w") as f:
        json.dump(sample_students, f, indent=2)
    
    # Create sample class mapping
    class_mapping = {
        "mapping": {
            "STU001": 0,
            "STU002": 1
        },
        "next_class_id": 2
    }
    
    with open("data/class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    print("✅ Created sample students data")
    print()

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    modules_to_test = [
        "tensorflow",
        "cv2", 
        "numpy",
        "flask",
        "firebase_admin",
        "mediapipe"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies manually")
        return False
    
    print("✅ All imports successful")
    print()
    return True

def run_training_demo():
    """Run a quick training demo"""
    print("Running training demo...")
    try:
        result = subprocess.run([
            sys.executable, "train_model.py",
            "--epochs", "5",
            "--num-samples", "20",
            "--num-classes", "2"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training demo completed successfully")
        else:
            print(f"⚠️  Training demo failed: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  Training demo error: {e}")
    print()

def print_next_steps():
    """Print next steps for the user"""
    print("=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Set up Firebase (optional for demo mode)")
    print("3. Run the system:")
    print("   python main.py --mode demo")
    print()
    print("Available modes:")
    print("  - demo: Run with simulated data")
    print("  - training: Train the model")
    print("  - production: Full system with real data")
    print()
    print("Web dashboard will be available at:")
    print("  http://localhost:5000")
    print()
    print("Default login credentials:")
    print("  Username: admin")
    print("  Password: admin123")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Create environment file
    create_env_file()
    
    # Create sample data
    create_sample_data()
    
    # Test imports
    if not test_imports():
        return 1
    
    # Run training demo
    run_training_demo()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 