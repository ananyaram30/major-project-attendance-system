# Intelligent Attendance Monitoring System

An automated, contactless attendance system using biometric **GAIT recognition** that identifies individuals based on unique walking patterns. This non-intrusive and hygienic solution is designed for educational institutions, workplaces, and secure facilities.

**Key Features:**
- **Model Training**: Upload training videos for each student to train the gait recognition model
- **Video Upload Processing**: Upload videos for batch attendance processing
- **CNN-based Gait Recognition**: Deep learning model for extracting unique gait features
- **Web Dashboard**: User-friendly interface for training and attendance monitoring
- **Firebase Integration**: Secure cloud-based attendance storage
- **Pose Estimation**: MediaPipe integration for enhanced accuracy
- **Multi-person Detection**: Support for multiple students in frame

## System Architecture

### Training Workflow
1. **Add Students**: Register students in the system
2. **Upload Training Videos**: Upload multiple videos of each student walking
3. **Train Model**: Use the training interface to train the CNN model
4. **Model Ready**: Once trained, the model can recognize students from new videos

### Attendance Processing Workflow
1. **Video Upload**: Upload videos containing students for attendance
2. **Video Processing**: System processes uploaded videos frame by frame
3. **Gait Analysis**: Trained CNN model extracts gait features and identifies students
4. **Attendance Detection**: Records attendance for recognized students
5. **Results Display**: View processing results and attendance statistics

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd intelligent-attendance-system

# Install dependencies
pip install -r requirements.txt

# Set up Firebase (see Firebase Setup section below)
python setup_firebase.py
```

### 2. Firebase Setup

1. **Create Firebase Project**:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project
   - Enable Realtime Database

2. **Download Service Account Key**:
   - Go to Project Settings > Service Accounts
   - Click "Generate new private key"
   - Save as `firebase-key.json` in project root

3. **Configure Database**:
   ```bash
   python setup_firebase.py
   ```

### 3. Run the System

```bash
# Start in demo mode (web dashboard)
python main.py --mode demo

# Start in training mode
python main.py --mode training
```

### 4. Access Web Dashboard

Open your browser and go to: `http://localhost:5000`

**Default Login:**
- Username: `admin`
- Password: `admin123`

## Usage Guide

### Step 1: Add Students
1. Go to **Students** page
2. Click **Add Student**
3. Fill in student details (ID, name, email, etc.)
4. Save the student

### Step 2: Upload Training Videos
1. Go to **Training** page
2. Click **Upload Training Video**
3. Select a student from the dropdown
4. Upload a video of the student walking
5. Repeat for multiple videos per student (recommended: 3-5 videos)

### Step 3: Train the Model
1. On the **Training** page, click **Start Training**
2. Wait for training to complete (progress will be shown)
3. Model will be saved and ready for use

### Step 4: Process Attendance Videos
1. Go to **Videos** page
2. Upload videos containing students for attendance
3. Click **Process** on uploaded videos
4. View results and attendance records

### Step 5: View Reports
1. Go to **Reports** page
2. Filter by date range and students
3. Export attendance data as CSV or JSON

## System Modes

### Demo Mode (`--mode demo`)
- Web dashboard for training and attendance processing
- No real-time camera detection
- Focus on video upload and processing

### Training Mode (`--mode training`)
- Same as demo mode but emphasizes training workflow
- Use for initial model training setup

## Configuration

Edit `.env` file to customize settings:

```env
# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=./firebase-key.json
FIREBASE_DATABASE_URL=https://your-project-id-default-rtdb.firebaseio.com/

# Model Configuration
CONFIDENCE_THRESHOLD=0.8
FRAME_BUFFER_SIZE=30
TEMPORAL_WINDOW=30

# Web Dashboard
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

## File Structure

```
intelligent-attendance-system/
├── attendance_system/          # Core attendance logic
├── data_processing/           # Video processing modules
├── models/                    # CNN models and pose estimation
├── web_dashboard/            # Flask web application
├── utils/                    # Utilities and helpers
├── data/                     # Data storage
│   ├── training_data/        # Training videos
│   ├── uploads/             # Uploaded videos for processing
│   ├── models/              # Trained models
│   └── reports/             # Generated reports
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## API Endpoints

### Training
- `GET /api/training/status` - Get training status
- `GET /api/training/videos` - Get training videos
- `POST /api/training/upload` - Upload training video
- `POST /api/training/start` - Start model training
- `DELETE /api/training/videos/<id>` - Delete training video

### Attendance
- `GET /api/attendance` - Get attendance records
- `POST /api/attendance` - Add attendance record
- `GET /api/attendance/export` - Export attendance data

### System
- `GET /api/system/status` - Get system status
- `GET /api/students` - Get students list
- `GET /api/statistics` - Get attendance statistics

## Troubleshooting

### Common Issues

1. **Firebase Connection Error (404)**:
   - Ensure Realtime Database is created in Firebase Console
   - Check `FIREBASE_DATABASE_URL` in `.env` file
   - Verify `firebase-key.json` is in project root

2. **Model Training Fails**:
   - Ensure training videos are uploaded
   - Check video format (MP4, AVI, MOV, MKV)
   - Verify students are added to the system

3. **Video Processing Errors**:
   - Check video file integrity
   - Ensure model is trained before processing
   - Verify video contains clear walking sequences

### Logs

Check logs in `./logs/attendance_system.log` for detailed error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs for error details 