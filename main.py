#!/usr/bin/env python3
"""
Intelligent Attendance Monitoring System
Main entry point for the application
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attendance_system.attendance_tracker import AttendanceTracker
from web_dashboard.app import app
from utils.config import Config
from utils.logger import logger

class IntelligentAttendanceSystem:
    """Main system class for the Intelligent Attendance Monitoring System"""
    
    def __init__(self):
        self.attendance_tracker = AttendanceTracker()
        self.web_server_thread = None
        self.is_running = False

    def start(self, mode: str = 'demo'):
        """Start the attendance system"""
        try:
            logger.info("=" * 60)
            logger.info("Intelligent Attendance Monitoring System")
            logger.info("=" * 60)
            
            if mode == 'demo':
                logger.info("Starting demo mode...")
                self._start_demo_mode()
            elif mode == 'training':
                logger.info("Starting training mode...")
                self._start_training_mode()
            else:
                logger.error(f"Unknown mode: {mode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            return False

    def _start_demo_mode(self):
        """Start demo mode with web dashboard"""
        try:
            # Initialize attendance tracker
            logger.info("Initializing Intelligent Attendance System...")
            if not self.attendance_tracker.initialize():
                logger.error("Failed to initialize attendance tracker")
                return False
            
            logger.info("System initialized successfully")
            
            # Start web server
            logger.info("Starting system in demo mode...")
            self._start_web_server()
            
        except Exception as e:
            logger.error(f"Error in demo mode: {e}")

    def _start_training_mode(self):
        """Start training mode"""
        try:
            logger.info("Training mode - use web dashboard to upload training videos and train model")
            
            # Initialize attendance tracker
            if not self.attendance_tracker.initialize():
                logger.error("Failed to initialize attendance tracker")
                return False
            
            # Start web server for training interface
            self._start_web_server()
            
        except Exception as e:
            logger.error(f"Error in training mode: {e}")

    def _start_web_server(self):
        """Start the web server"""
        try:
            logger.info(f"Web server started on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
            logger.info("System is running. Press Ctrl+C to stop.")
            
            # Start system health monitoring
            self._start_health_monitoring()
            
            # Run the Flask app
            app.run(
                host=Config.FLASK_HOST,
                port=Config.FLASK_PORT,
                debug=Config.FLASK_DEBUG,
                use_reloader=False  # Disable reloader to avoid duplicate processes
            )
            
        except Exception as e:
            logger.error(f"Error starting web server: {e}")

    def _start_health_monitoring(self):
        """Start system health monitoring"""
        import threading
        import time
        
        def monitor_health():
            while self.is_running:
                try:
                    # Get system status
                    status = self.attendance_tracker.get_system_status()
                    
                    # Log performance metrics
                    logger.info(f"Performance - Total Detections: {self.attendance_tracker.detection_count}, "
                               f"Total Attendance: {self.attendance_tracker.attendance_count}, "
                               f"Avg Processing Time: {self.attendance_tracker._get_average_processing_time():.3f}s")
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error monitoring system health: {e}")
                    time.sleep(30)
        
        self.is_running = True
        health_thread = threading.Thread(target=monitor_health, daemon=True)
        health_thread.start()

    def stop(self):
        """Stop the attendance system"""
        try:
            logger.info("Stopping Intelligent Attendance System...")
            self.is_running = False
            
            if self.attendance_tracker:
                self.attendance_tracker.stop_tracking()
            
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")

def main():
    """Main entry point - Video Upload & Training System"""
    parser = argparse.ArgumentParser(description='Intelligent Attendance System - Video Upload & Training')
    parser.add_argument('--mode', choices=['web'], default='web',
                       help='Run mode: web for video upload and training interface')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting Intelligent Attendance System - Video Upload Mode")
        logger.info("Access the web interface to upload videos and train the model")
        
        # Start web interface only (no real-time processing)
        from web_dashboard.app import app
        
        logger.info(f"Web server starting on http://127.0.0.1:{Config.FLASK_PORT}")
        logger.info("Open your browser and go to: http://127.0.0.1:5000")
        logger.info("Login with: admin / admin123")
        
        app.run(
            host='127.0.0.1',
            port=Config.FLASK_PORT,
            debug=False,
            use_reloader=False
        )
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Shutting down web server...")
        return True
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        return False

if __name__ == "__main__":
    main() 