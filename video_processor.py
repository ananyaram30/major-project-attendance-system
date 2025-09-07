import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator, Dict
from collections import deque
import threading
import time
import os
from datetime import datetime
import json
from utils.config import Config
from utils.logger import logger
from utils.helpers import ImageProcessor, performance_monitor

class VideoProcessor:
    """Real-time video processing for gait recognition"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)
        self.processing_thread = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        logger.info(f"Video processor initialized with camera index {camera_index}")
    
    def start_camera(self) -> bool:
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from camera"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input"""
        start_time = performance_monitor.start_timer()
        
        # Resize frame
        resized = ImageProcessor.resize_image(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
        
        # Normalize
        normalized = ImageProcessor.normalize_image(resized)
        
        processing_time = performance_monitor.end_timer(start_time)
        performance_monitor.log_performance(processing_time)
        
        return normalized
    
    def extract_frame_sequence(self, num_frames: int = Config.TEMPORAL_WINDOW) -> List[np.ndarray]:
        """Extract sequence of frames for temporal analysis"""
        with self.lock:
            frames = list(self.frame_buffer)[-num_frames:]
        
        # Pad with last frame if not enough frames
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                # Create empty frame if no frames available
                empty_frame = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
                frames.append(empty_frame)
        
        return frames
    
    def process_frame_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """Process sequence of frames for model input"""
        processed_frames = []
        
        for frame in frames:
            processed = self.preprocess_frame(frame)
            processed_frames.append(processed)
        
        # Stack frames for 3D CNN or temporal model
        sequence = np.stack(processed_frames, axis=0)
        return np.expand_dims(sequence, axis=0)
    
    def start_processing(self):
        """Start real-time frame processing"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Frame processing started")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            frame = self.read_frame()
            if frame is not None:
                with self.lock:
                    self.frame_buffer.append(frame)
                
                # Update FPS
                self._update_fps()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
        return None
    
    def get_frame_sequence(self, length: int = Config.TEMPORAL_WINDOW) -> List[np.ndarray]:
        """Get sequence of recent frames"""
        with self.lock:
            frames = list(self.frame_buffer)[-length:]
        
        return frames
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps

class VideoUploadProcessor:
    """Process uploaded video files for attendance tracking"""
    
    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = upload_dir
        self.processed_dir = "./data/processed"
        self.results_dir = "./data/results"
        
        # Create directories
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Supported video formats
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        logger.info(f"Video upload processor initialized with upload directory: {upload_dir}")
    
    def get_uploaded_videos(self) -> List[Dict]:
        """Get list of uploaded videos"""
        videos = []
        
        if not os.path.exists(self.upload_dir):
            return videos
        
        for filename in os.listdir(self.upload_dir):
            filepath = os.path.join(self.upload_dir, filename)
            if os.path.isfile(filepath):
                # Check if it's a supported video format
                _, ext = os.path.splitext(filename)
                if ext.lower() in self.supported_formats:
                    video_info = self._get_video_info(filepath)
                    # Ensure all required fields are present
                    video_data = {
                        'filename': filename,
                        'filepath': filepath,
                        'status': self._get_processing_status(filename),
                        'fps': video_info.get('fps', 0),
                        'frame_count': video_info.get('frame_count', 0),
                        'width': video_info.get('width', 0),
                        'height': video_info.get('height', 0),
                        'duration': video_info.get('duration', 0),
                        'file_size_mb': video_info.get('file_size_mb', 0),
                        'upload_time': video_info.get('upload_time', '')
                    }
                    videos.append(video_data)
        
        return videos
    
    def _get_video_info(self, filepath: str) -> Dict:
        """Get video file information"""
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            file_size = os.path.getsize(filepath)
            upload_time = datetime.fromtimestamp(os.path.getctime(filepath))
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size_mb': file_size / (1024 * 1024),
                'upload_time': upload_time.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting video info for {filepath}: {e}")
            return {}
    
    def _get_processing_status(self, filename: str) -> str:
        """Get processing status of video file"""
        base_name = os.path.splitext(filename)[0]
        result_file = os.path.join(self.results_dir, f"{base_name}_results.json")
        
        if os.path.exists(result_file):
            return "processed"
        else:
            return "pending"
    
    def process_video(self, filename: str, callback=None) -> Dict:
        """Process uploaded video file for attendance tracking"""
        filepath = os.path.join(self.upload_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        logger.info(f"Starting processing of video: {filename}")
        
        # Initialize video processor
        video_processor = VideoFileProcessor(filepath)
        
        if not video_processor.open_video():
            raise ValueError(f"Failed to open video file: {filepath}")
        
        try:
            # Get video info
            video_info = video_processor.get_video_info()
            
            # Process video frames
            attendance_results = self._process_video_frames(video_processor, callback)
            
            # Save results
            results = {
                'filename': filename,
                'video_info': video_info,
                'attendance_results': attendance_results,
                'processing_time': time.time(),
                'total_detections': len(attendance_results),
                'unique_students': len(set(result['student_id'] for result in attendance_results))
            }
            
            # Save to results directory
            base_name = os.path.splitext(filename)[0]
            result_file = os.path.join(self.results_dir, f"{base_name}_results.json")
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Video processing completed: {filename}")
            return results
            
        finally:
            video_processor.close_video()
    
    def _process_video_frames(self, video_processor: 'VideoFileProcessor', callback=None) -> List[Dict]:
        """Process video frames for attendance detection"""
        attendance_results = []
        frame_count = 0
        
        # Get total frames for progress tracking
        total_frames = video_processor.get_video_info().get('frame_count', 0)
        
        # Extract frames in batches
        batch_size = Config.TEMPORAL_WINDOW
        frames = []
        
        while True:
            frame = video_processor.read_frame()
            if frame is None:
                break
            
            frames.append(frame)
            frame_count += 1
            
            # Process batch when we have enough frames
            if len(frames) >= batch_size:
                batch_results = self._process_frame_batch(frames)
                attendance_results.extend(batch_results)
                
                # Keep last few frames for overlap
                frames = frames[-Config.TEMPORAL_WINDOW//2:]
            
            # Progress callback
            if callback and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                callback(progress)
        
        # Process remaining frames
        if frames:
            batch_results = self._process_frame_batch(frames)
            attendance_results.extend(batch_results)
        
        return attendance_results
    
    def _process_frame_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """Process a batch of frames for attendance detection"""
        # This would integrate with the attendance tracking system
        # For now, return empty results
        return []
    
    def delete_video(self, filename: str) -> bool:
        """Delete uploaded video file"""
        try:
            filepath = os.path.join(self.upload_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted video file: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting video file {filename}: {e}")
            return False
    
    def get_processing_results(self, filename: str) -> Optional[Dict]:
        """Get processing results for a video file"""
        try:
            base_name = os.path.splitext(filename)[0]
            result_file = os.path.join(self.results_dir, f"{base_name}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading results for {filename}: {e}")
            return None
    
    def _save_processing_results(self, filename: str, results: Dict):
        """Save processing results for a video file"""
        try:
            base_name = os.path.splitext(filename)[0]
            result_file = os.path.join(self.results_dir, f"{base_name}_results.json")
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Processing results saved for {filename}")
        except Exception as e:
            logger.error(f"Error saving results for {filename}: {e}")

class VideoFileProcessor:
    """Process video files for training and testing"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        
    def open_video(self) -> bool:
        """Open video file"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.video_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error opening video file: {e}")
            return False
    
    def close_video(self):
        """Close video file"""
        if self.cap:
            self.cap.release()
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def get_video_info(self) -> dict:
        """Get video file information"""
        if not self.cap:
            return {}
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def extract_frames(self, start_frame: int = 0, end_frame: int = None, 
                      sample_rate: int = 1) -> List[np.ndarray]:
        """Extract frames from video file"""
        frames = []
        
        if not self.cap:
            return frames
        
        if end_frame is None:
            end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if (frame_idx - start_frame) % sample_rate == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        return frames
    
    def extract_frame_sequences(self, sequence_length: int = Config.TEMPORAL_WINDOW,
                              overlap: float = 0.5) -> List[List[np.ndarray]]:
        """Extract overlapping frame sequences"""
        sequences = []
        
        if not self.cap:
            return sequences
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_size = int(sequence_length * (1 - overlap))
        
        for start_frame in range(0, total_frames - sequence_length + 1, step_size):
            sequence = self.extract_frames(start_frame, start_frame + sequence_length)
            if len(sequence) == sequence_length:
                sequences.append(sequence)
        
        return sequences

class FrameProcessor:
    """Advanced frame processing utilities"""
    
    @staticmethod
    def detect_motion(frames: List[np.ndarray], threshold: float = 0.1) -> bool:
        """Detect motion between frames"""
        if len(frames) < 2:
            return False
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # Calculate frame differences
        diff = cv2.absdiff(gray_frames[0], gray_frames[-1])
        
        # Calculate motion percentage
        motion_percentage = np.sum(diff > 25) / diff.size
        
        return motion_percentage > threshold
    
    @staticmethod
    def stabilize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
        """Stabilize frame sequence"""
        if len(frames) < 2:
            return frames
        
        stabilized_frames = [frames[0]]
        
        for i in range(1, len(frames)):
            # Calculate optical flow
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Apply stabilization
            h, w = flow.shape[:2]
            flow_map = np.column_stack((flow.reshape(-1, 2), np.ones((h*w, 1))))
            
            # Warp frame
            stabilized = cv2.warpAffine(frames[i], flow_map[:2, :].T, (w, h))
            stabilized_frames.append(stabilized)
        
        return stabilized_frames
    
    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def remove_background(frames: List[np.ndarray]) -> List[np.ndarray]:
        """Remove background from frame sequence"""
        if len(frames) < 3:
            return frames
        
        # Create background model
        background_model = np.median(frames, axis=0).astype(np.uint8)
        
        # Remove background from each frame
        foreground_frames = []
        for frame in frames:
            diff = cv2.absdiff(frame, background_model)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply mask to frame
            foreground = cv2.bitwise_and(frame, mask)
            foreground_frames.append(foreground)
        
        return foreground_frames 