import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils.config import Config
from utils.logger import logger

class PoseEstimator:
    """MediaPipe-based pose estimation for gait analysis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=Config.POSE_CONFIDENCE,
            min_tracking_confidence=Config.POSE_SMOOTHING
        )
        
        # Define key pose landmarks for gait analysis
        self.gait_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE,
            self.mp_pose.PoseLandmark.RIGHT_EYE
        ]
        
        logger.info("Pose estimator initialized")
    
    def detect_pose(self, frame: np.ndarray) -> Optional[mp.solutions.pose.Pose]:
        """Detect pose in a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        return results
    
    def extract_pose_features(self, pose_results: mp.solutions.pose.Pose) -> Dict:
        """Extract pose features for gait analysis"""
        if not pose_results.pose_landmarks:
            return None
        
        landmarks = pose_results.pose_landmarks.landmark
        features = {}
        
        # Extract key landmark coordinates
        for landmark_id in self.gait_landmarks:
            landmark = landmarks[landmark_id.value]
            features[f"landmark_{landmark_id.name}_x"] = landmark.x
            features[f"landmark_{landmark_id.name}_y"] = landmark.y
            features[f"landmark_{landmark_id.name}_z"] = landmark.z
            features[f"landmark_{landmark_id.name}_visibility"] = landmark.visibility
        
        # Calculate pose-based features
        features.update(self._calculate_pose_angles(landmarks))
        features.update(self._calculate_pose_distances(landmarks))
        features.update(self._calculate_pose_velocities(landmarks))
        
        return features
    
    def _calculate_pose_angles(self, landmarks: List) -> Dict:
        """Calculate angles between key body parts"""
        angles = {}
        
        # Shoulder angle
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Calculate shoulder angles
        left_shoulder_angle = self._calculate_angle(
            [left_shoulder.x, left_shoulder.y],
            [left_elbow.x, left_elbow.y],
            [right_shoulder.x, right_shoulder.y]
        )
        right_shoulder_angle = self._calculate_angle(
            [right_shoulder.x, right_shoulder.y],
            [right_elbow.x, right_elbow.y],
            [left_shoulder.x, left_shoulder.y]
        )
        
        angles['left_shoulder_angle'] = left_shoulder_angle
        angles['right_shoulder_angle'] = right_shoulder_angle
        
        # Hip angles
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        left_hip_angle = self._calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [right_hip.x, right_hip.y]
        )
        right_hip_angle = self._calculate_angle(
            [right_hip.x, right_hip.y],
            [right_knee.x, right_knee.y],
            [left_hip.x, left_hip.y]
        )
        
        angles['left_hip_angle'] = left_hip_angle
        angles['right_hip_angle'] = right_hip_angle
        
        return angles
    
    def _calculate_pose_distances(self, landmarks: List) -> Dict:
        """Calculate distances between key body parts"""
        distances = {}
        
        # Shoulder width
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_width = self._calculate_distance(left_shoulder, right_shoulder)
        distances['shoulder_width'] = shoulder_width
        
        # Hip width
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_width = self._calculate_distance(left_hip, right_hip)
        distances['hip_width'] = hip_width
        
        # Torso height
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        mid_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]  # Use left hip as reference
        torso_height = self._calculate_distance(nose, mid_hip)
        distances['torso_height'] = torso_height
        
        return distances
    
    def _calculate_pose_velocities(self, landmarks: List) -> Dict:
        """Calculate pose velocities (placeholder for temporal analysis)"""
        velocities = {}
        
        # This would require tracking landmarks over time
        # For now, return empty dict
        # In a full implementation, you would track previous frames
        # and calculate velocities based on position changes
        
        return velocities
    
    def _calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_distance(self, point1, point2) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def draw_pose_landmarks(self, frame: np.ndarray, pose_results: mp.solutions.pose.Pose) -> np.ndarray:
        """Draw pose landmarks on frame"""
        annotated_frame = frame.copy()
        
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return annotated_frame
    
    def extract_gait_features(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Extract gait features from frame with pose estimation"""
        # Detect pose
        pose_results = self.detect_pose(frame)
        
        if not pose_results or not pose_results.pose_landmarks:
            return None, frame
        
        # Extract pose features
        pose_features = self.extract_pose_features(pose_results)
        
        # Draw landmarks
        annotated_frame = self.draw_pose_landmarks(frame, pose_results)
        
        return pose_features, annotated_frame
    
    def process_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """Process a sequence of frames and extract temporal pose features"""
        pose_features_sequence = []
        
        for frame in frames:
            pose_features, _ = self.extract_gait_features(frame)
            if pose_features:
                pose_features_sequence.append(pose_features)
        
        return pose_features_sequence
    
    def get_pose_confidence(self, pose_results: mp.solutions.pose.Pose) -> float:
        """Get overall pose detection confidence"""
        if not pose_results.pose_landmarks:
            return 0.0
        
        landmarks = pose_results.pose_landmarks.landmark
        visibilities = [landmark.visibility for landmark in landmarks]
        return np.mean(visibilities)
    
    def is_pose_valid(self, pose_results: mp.solutions.pose.Pose) -> bool:
        """Check if detected pose is valid for gait analysis"""
        if not pose_results.pose_landmarks:
            return False
        
        confidence = self.get_pose_confidence(pose_results)
        return confidence > Config.POSE_CONFIDENCE
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        self.pose.close()
        logger.info("Pose estimator cleaned up")

class PoseFeatureExtractor:
    """Advanced pose feature extraction for gait analysis"""
    
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.previous_features = None
    
    def extract_advanced_features(self, frame: np.ndarray) -> Dict:
        """Extract advanced pose-based gait features"""
        pose_features, annotated_frame = self.pose_estimator.extract_gait_features(frame)
        
        if not pose_features:
            return None
        
        # Add temporal features if previous features exist
        if self.previous_features:
            temporal_features = self._calculate_temporal_features(pose_features, self.previous_features)
            pose_features.update(temporal_features)
        
        # Update previous features
        self.previous_features = pose_features.copy()
        
        return pose_features
    
    def _calculate_temporal_features(self, current_features: Dict, previous_features: Dict) -> Dict:
        """Calculate temporal features between current and previous pose"""
        temporal_features = {}
        
        # Calculate velocity for key landmarks
        for key in current_features:
            if key in previous_features and isinstance(current_features[key], (int, float)):
                velocity = current_features[key] - previous_features[key]
                temporal_features[f"{key}_velocity"] = velocity
        
        return temporal_features
    
    def extract_multi_person_features(self, frame: np.ndarray) -> List[Dict]:
        """Extract features for multiple people in frame"""
        # This would require multi-person pose detection
        # For now, return single person features
        features = self.extract_advanced_features(frame)
        return [features] if features else []
    
    def get_feature_vector(self, pose_features: Dict) -> np.ndarray:
        """Convert pose features to feature vector"""
        if not pose_features:
            return np.array([])
        
        # Convert features to numerical vector
        feature_vector = []
        for key, value in pose_features.items():
            if isinstance(value, (int, float)):
                feature_vector.append(value)
        
        return np.array(feature_vector) 