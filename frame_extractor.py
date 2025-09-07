import os
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from datetime import datetime
import json

from utils.logger import logger
from utils.config import Config


class StudentFrameExtractor:
    """Enhanced frame extractor for individual student video processing"""
    
    def __init__(self):
        self.base_frames_dir = os.path.join(Config.DATA_DIR, "frames")
        self.training_frames_dir = os.path.join(Config.TRAINING_DATA_DIR, "frames")
        os.makedirs(self.base_frames_dir, exist_ok=True)
        os.makedirs(self.training_frames_dir, exist_ok=True)
    
    def extract_student_training_frames(self, video_path: str, student_id: str, 
                                      video_type: str = "training") -> Dict:
        """
        Extract frames from student training video
        
        Args:
            video_path: Path to the video file
            student_id: Student ID for organizing frames
            video_type: Type of video (training/attendance)
            
        Returns:
            Dictionary with extraction results
        """
        try:
            logger.info(f"Starting frame extraction for student {student_id}")
            
            # Create student-specific directory
            if video_type == "training":
                output_dir = os.path.join(self.training_frames_dir, student_id)
            else:
                output_dir = os.path.join(self.base_frames_dir, student_id)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract frames
            extraction_result = self._extract_frames_with_metadata(
                video_path, output_dir, student_id, video_type
            )
            
            logger.info(f"Frame extraction completed for student {student_id}")
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error extracting frames for student {student_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'student_id': student_id,
                'frames_extracted': 0
            }
    
    def _extract_frames_with_metadata(self, video_path: str, output_dir: str, 
                                    student_id: str, video_type: str) -> Dict:
        """Extract frames with detailed metadata"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Calculate sampling rate based on video duration
        sample_rate = self._calculate_sample_rate(duration, total_frames)
        
        frame_index = 0
        saved_count = 0
        extracted_frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on calculated rate
                if frame_index % sample_rate == 0:
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    
                    # Generate frame filename with timestamp
                    timestamp = frame_index / fps if fps > 0 else frame_index
                    frame_filename = f"{student_id}_frame_{frame_index:06d}_t{timestamp:.2f}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Save frame
                    cv2.imwrite(frame_path, processed_frame)
                    
                    # Store frame metadata
                    frame_info = {
                        'filename': frame_filename,
                        'path': frame_path,
                        'frame_index': frame_index,
                        'timestamp': timestamp,
                        'width': processed_frame.shape[1],
                        'height': processed_frame.shape[0]
                    }
                    extracted_frames.append(frame_info)
                    saved_count += 1
                
                frame_index += 1
            
            # Save extraction metadata
            metadata = {
                'student_id': student_id,
                'video_type': video_type,
                'video_path': video_path,
                'output_dir': output_dir,
                'extraction_date': datetime.now().isoformat(),
                'video_properties': {
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration': duration,
                    'width': width,
                    'height': height
                },
                'extraction_settings': {
                    'sample_rate': sample_rate,
                    'frames_extracted': saved_count,
                    'preprocessing_applied': True
                },
                'extracted_frames': extracted_frames
            }
            
            # Save metadata file
            metadata_path = os.path.join(output_dir, f"{student_id}_extraction_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Extracted {saved_count} frames for student {student_id}")
            
            return {
                'success': True,
                'student_id': student_id,
                'frames_extracted': saved_count,
                'output_dir': output_dir,
                'metadata_path': metadata_path,
                'video_duration': duration,
                'sample_rate': sample_rate,
                'extracted_frames': extracted_frames
            }
            
        finally:
            cap.release()
    
    def _calculate_sample_rate(self, duration: float, total_frames: int) -> int:
        """Calculate optimal sampling rate based on video duration"""
        
        # Target: Extract 30-60 frames per video for training
        target_frames = 45
        
        if total_frames <= target_frames:
            return 1  # Extract all frames
        
        sample_rate = max(1, total_frames // target_frames)
        logger.info(f"Calculated sample rate: {sample_rate} (extracting every {sample_rate}th frame)")
        return sample_rate
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better training quality"""
        
        # Resize to standard size
        target_size = (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        frame = cv2.resize(frame, target_size)
        
        # Apply basic enhancement
        # Convert to LAB color space for better lighting normalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def extract_attendance_frames(self, video_path: str, student_id: str = None) -> Dict:
        """Extract frames from attendance video for recognition"""
        
        try:
            logger.info(f"Extracting frames from attendance video: {video_path}")
            
            # Create attendance frames directory
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(self.base_frames_dir, "attendance", video_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract frames for attendance checking
            result = self._extract_frames_with_metadata(
                video_path, output_dir, student_id or "unknown", "attendance"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting attendance frames: {e}")
            return {
                'success': False,
                'error': str(e),
                'frames_extracted': 0
            }
    
    def get_student_frames(self, student_id: str, frame_type: str = "training") -> List[str]:
        """Get list of extracted frames for a student"""
        
        if frame_type == "training":
            frames_dir = os.path.join(self.training_frames_dir, student_id)
        else:
            frames_dir = os.path.join(self.base_frames_dir, student_id)
        
        if not os.path.exists(frames_dir):
            return []
        
        frame_files = [
            os.path.join(frames_dir, f) 
            for f in os.listdir(frames_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        return sorted(frame_files)
    
    def get_extraction_metadata(self, student_id: str, frame_type: str = "training") -> Dict:
        """Get extraction metadata for a student"""
        
        if frame_type == "training":
            frames_dir = os.path.join(self.training_frames_dir, student_id)
        else:
            frames_dir = os.path.join(self.base_frames_dir, student_id)
        
        metadata_path = os.path.join(frames_dir, f"{student_id}_extraction_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return {}


# Legacy function for backward compatibility
def save_video_frames(video_path: str, output_dir: Optional[str] = None, sample_rate: int = 1) -> str:
    """
    Legacy function - Extract frames from a video and save them to a directory.
    """
    extractor = StudentFrameExtractor()
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_output_dir = output_dir or os.path.join("./data/frames", video_name)
    os.makedirs(base_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frame_index = 0
    saved_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if sample_rate <= 1 or (frame_index % sample_rate == 0):
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_path = os.path.join(base_output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_index += 1

        logger.info(f"Saved {saved_count} frames to {base_output_dir}")
        return base_output_dir
    finally:
        cap.release()