#!/usr/bin/env python3
"""
Database Connection Test Script
for Intelligent Attendance Monitoring System
"""

import os
import sys
from datetime import datetime

def test_database_connection():
    """Test database connection and basic operations"""
    try:
        from attendance_system.database_handler import DatabaseHandler
        
        print("Testing database connection...")
        
        # Initialize database handler
        db_handler = DatabaseHandler()
        
        # Test initialization
        if not db_handler.initialize():
            print("❌ Failed to initialize database")
            return False
        
        print("✅ Database initialized successfully")
        
        # Test adding a sample student
        sample_student = {
            'student_id': 'TEST001',
            'name': 'Test Student',
            'email': 'test@example.com',
            'department': 'Computer Science',
            'year': '2024'
        }
        
        if db_handler.add_student(sample_student):
            print("✅ Sample student added successfully")
        else:
            print("❌ Failed to add sample student")
        
        # Test adding attendance record
        sample_attendance = {
            'student_id': 'TEST001',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.95,
            'source': 'test'
        }
        
        if db_handler.add_attendance_record(sample_attendance):
            print("✅ Sample attendance record added successfully")
        else:
            print("❌ Failed to add attendance record")
        
        # Test retrieving data
        students = db_handler.get_all_students()
        print(f"✅ Retrieved {len(students)} students from database")
        
        attendance_records = db_handler.get_attendance_records()
        print(f"✅ Retrieved {len(attendance_records)} attendance records from database")
        
        # Clean up test data
        db_handler.delete_student('TEST001')
        print("✅ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_student_manager():
    """Test student manager functionality"""
    try:
        from attendance_system.student_manager import StudentManager
        
        print("\nTesting student manager...")
        
        # Initialize student manager
        student_manager = StudentManager()
        
        # Test adding student
        test_student = {
            'student_id': 'TEST002',
            'name': 'Test Student 2',
            'email': 'test2@example.com',
            'department': 'Engineering',
            'year': '2024'
        }
        
        if student_manager.add_student(test_student):
            print("✅ Student manager: Student added successfully")
        else:
            print("❌ Student manager: Failed to add student")
        
        # Test retrieving student
        student = student_manager.get_student('TEST002')
        if student:
            print("✅ Student manager: Student retrieved successfully")
        else:
            print("❌ Student manager: Failed to retrieve student")
        
        # Test updating student
        updates = {'department': 'Computer Engineering'}
        if student_manager.update_student('TEST002', updates):
            print("✅ Student manager: Student updated successfully")
        else:
            print("❌ Student manager: Failed to update student")
        
        # Test deleting student
        if student_manager.delete_student('TEST002'):
            print("✅ Student manager: Student deleted successfully")
        else:
            print("❌ Student manager: Failed to delete student")
        
        return True
        
    except Exception as e:
        print(f"❌ Student manager test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Database Connection Test")
    print("Intelligent Attendance Monitoring System")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please run setup_firebase.py first.")
        return
    
    # Check if Firebase key exists
    if not os.path.exists('./firebase-key.json'):
        print("❌ firebase-key.json not found. Please download your Firebase service account key.")
        return
    
    # Test database connection
    db_success = test_database_connection()
    
    # Test student manager
    student_success = test_student_manager()
    
    print("\n" + "=" * 60)
    if db_success and student_success:
        print("✅ ALL DATABASE TESTS PASSED!")
        print("Your database is properly configured and working.")
        print()
        print("You can now run the system with:")
        print("  python main.py --mode demo")
    else:
        print("❌ SOME DATABASE TESTS FAILED!")
        print("Please check your Firebase configuration.")
        print()
        print("Common issues:")
        print("1. Firebase key file is invalid or corrupted")
        print("2. Database URL is incorrect")
        print("3. Network connectivity issues")
        print("4. Firebase project not properly configured")
    print("=" * 60)

if __name__ == "__main__":
    main() 