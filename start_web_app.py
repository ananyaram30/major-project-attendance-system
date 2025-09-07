#!/usr/bin/env python3
"""
Video Upload & Training System - Simple startup script
Upload videos and train AI model on-demand
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    try:
        print("=" * 60)
        print("Intelligent Attendance System - Video Upload & Training")
        print("=" * 60)
        print("Features:")
        print("• Upload student videos for training")
        print("• Train AI model when you click 'Train'")
        print("• No real-time processing - video-only workflow")
        print("=" * 60)
        
        # Import and start the Flask app
        from web_dashboard.app import app
        
        print("✓ Flask app loaded successfully")
        print("✓ Starting web server...")
        print()
        print("🌐 Open your browser and go to: http://127.0.0.1:5000")
        print("🔑 Login with: admin / admin123")
        print()
        print("Workflow:")
        print("1. Add students in 'Students' section")
        print("2. Upload training videos in 'Training' section")
        print("3. Click 'Start Training' to train the AI model")
        print("4. Upload test videos to check attendance")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the Flask development server
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Please install dependencies:")
        print("python -m pip install numpy python-dotenv flask opencv-python tensorflow mediapipe firebase-admin pandas scikit-learn matplotlib seaborn requests tqdm imutils psutil flask-cors")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
