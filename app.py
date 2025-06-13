import cv2
import torch
import numpy as np
import os
import sys
import pickle
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
from typing import List
import io
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime, timedelta
import heapq
import traceback
import logging
import tempfile
import subprocess
import shutil
import socket
import ffmpeg
import asyncio
import platform
import signal
import uvicorn
from uvicorn.config import Config
from ultralytics import YOLO
import json
from PIL import Image
import uuid
import math
from firebase_config import initialize_firebase
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase
try:
    firebase_app, firestore_db, storage_bucket = initialize_firebase()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    logger.error(traceback.format_exc())

# Default location (Coimbatore, India)
DEFAULT_LATITUDE = 11.016234
DEFAULT_LONGITUDE = 76.980540

# Configure event loop policy for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Global shutdown event
shutdown_event = asyncio.Event()

# Global status updates queue
status_updates_queue = asyncio.Queue()

# Initialize models
MODEL_PATH = "models/yolov8s.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize YOLO model for webcam
try:
    webcam_model = YOLO(MODEL_PATH)
    webcam_model.to(device)
    logger.info(f"Loaded webcam model on {device}")
except Exception as e:
    logger.error(f"Error loading webcam model: {str(e)}")
    logger.error(traceback.format_exc())
    webcam_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    logger.info("Starting up the application...")
    yield
    logger.info("Shutting down the application...")
    shutdown_event.set()

app = FastAPI(lifespan=lifespan)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal. Cleaning up...")
    shutdown_event.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Get the local hostname
        hostname = socket.gethostname()
        # Get the IP address
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        logger.error(f"Error getting local IP: {str(e)}")
        return "127.0.0.1"

# Get local IP address
LOCAL_IP = get_local_ip()
logger.info(f"Local IP address: {LOCAL_IP}")

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]
)

# Configure for large file uploads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Configure upload limits
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB in bytes

class CustomUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatch[pickle.PERSID] = self.persistent_load

    def persistent_load(self, pid):
        """Handle persistent ID loading."""
        # Just return None for any persistent ID
        return None

    def find_class(self, module, name):
        """Custom find_class that handles missing modules."""
        try:
            # Handle torch modules
            if module.startswith('torch.nn.modules'):
                return getattr(torch.nn.modules, name)
            if module.startswith('torch'):
                return getattr(torch, name)
            
            # Handle ultralytics modules by returning dummy objects
            if module.startswith('ultralytics'):
                return object
            
            # Try normal import
            return super().find_class(module, name)
        except Exception:
            # If anything fails, return a dummy object
            return object

def load_model_safe(path, map_location=None):
    """Safely load a PyTorch model with custom unpickling."""
    try:
        # First try direct loading with torch
        try:
            weights = torch.load(path, map_location=map_location, pickle_module=pickle)
        except Exception as e:
            logger.info(f"Direct torch.load failed, trying custom unpickler: {str(e)}")
            # If that fails, try with custom unpickler
            with open(path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                weights = unpickler.load()

        # Extract and process the state dict
        if isinstance(weights, dict):
            # Try different common model dictionary structures
            if 'model' in weights and isinstance(weights['model'], dict):
                weights = weights['model']
            elif 'state_dict' in weights:
                weights = weights['state_dict']
            elif 'model_state_dict' in weights:
                weights = weights['model_state_dict']
            
            # Convert to float32 and remove problematic keys
            processed_weights = {}
            for k, v in weights.items():
                # Remove 'model.' prefix if present
                key = k.replace('model.', '')
                # Only keep tensor values and convert to float32
                if isinstance(v, torch.Tensor):
                    processed_weights[key] = v.float()
            
            return processed_weights
        else:
            logger.warning("Loaded weights are not in expected dictionary format")
            return weights

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOv5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Basic backbone
        self.backbone = torch.nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512),
            ConvBlock(512, 1024, stride=2),
        )
        # Detection head
        self.head = torch.nn.Conv2d(1024, 3 * (5 + 1), 1)  # 3 anchors, 5 box params + 1 class
        self.stride = None
        self.names = None

    def _load_state_dict(self, state_dict):
        """Load state dict into model."""
        try:
            # Extract weights from the loaded dictionary
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Create new state dict without problematic keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):  # Only keep tensor values
                        key = k.replace('model.', '').replace('module.', '')
                        new_state_dict[key] = v.float()  # Convert to float32
                
                self.load_state_dict(new_state_dict, strict=False)
                logger.info("Model weights loaded successfully")
            else:
                logger.error("Unexpected model format")
                raise ValueError("Unexpected model format")
        except Exception as e:
            logger.error(f"Error loading state dict: {str(e)}")
            raise

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

def load_yolov5_weights(model_path, device='cpu'):
    """Load YOLOv5 weights with custom loading function."""
    try:
        # Define a custom loading function
        def custom_load(f):
            try:
                return torch.load(f, map_location=device)
            except Exception as e:
                # If direct loading fails, try loading with pickle
                if hasattr(f, 'seek'):
                    f.seek(0)
                return pickle.load(f)

        # Try different loading methods
        try:
            # Method 1: Direct loading
            weights = torch.load(model_path, map_location=device)
        except Exception:
            # Method 2: Custom loading with pickle
            with open(model_path, 'rb') as f:
                weights = custom_load(f)

        return weights
    except Exception as e:
        logger.error(f"Error in load_yolov5_weights: {str(e)}")
        raise

# Define vehicle classes (COCO dataset indices)
VEHICLE_CLASSES = {
    2: 'car',      # car
    5: 'bus',      # bus
    3: 'motorbike', # motorcycle
    7: 'truck'     # truck
}

# Define model paths
WEBCAM_MODEL_PATH = os.path.join("models", "uyir.pt")  # Previously yolov8s.pt
FILE_UPLOAD_MODEL_PATH = os.path.join("models", "uyir1.pt")  # Previously best.pt

# Load YOLOv8 models
if not os.path.exists(WEBCAM_MODEL_PATH):
    logger.error(f"Webcam model not found at: {WEBCAM_MODEL_PATH}")
    raise FileNotFoundError(f"Webcam model not found at: {WEBCAM_MODEL_PATH}")

if not os.path.exists(FILE_UPLOAD_MODEL_PATH):
    logger.error(f"File upload model not found at: {FILE_UPLOAD_MODEL_PATH}")
    raise FileNotFoundError(f"File upload model not found at: {FILE_UPLOAD_MODEL_PATH}")

try:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load both models
    logger.info(f"Loading webcam model from: {WEBCAM_MODEL_PATH}")
    webcam_model = YOLO(WEBCAM_MODEL_PATH)
    webcam_model.to(device)
    
    logger.info(f"Loading file upload model from: {FILE_UPLOAD_MODEL_PATH}")
    file_upload_model = YOLO(FILE_UPLOAD_MODEL_PATH)
    file_upload_model.to(device)
    
    # Force models to use CUDA if available
    if torch.cuda.is_available():
        webcam_model.cuda()
        file_upload_model.cuda()
        logger.info("Models moved to CUDA")
    
    logger.info(f"Models loaded successfully on {device}")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    raise RuntimeError(f"Failed to load models: {str(e)}")

def preprocess_image(frame):
    """Preprocess image for YOLOv8 model."""
    try:
        # Resize and normalize the image
        img = cv2.resize(frame, (640, 640))  # YOLOv8 default size
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0  # Normalize to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img.to(device)
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return None

def detect_accidents(frame, source_type='webcam'):
    """Perform accident detection on a single frame."""
    try:
        # Convert bytes to numpy array if input is bytes
        if isinstance(frame, bytes):
            nparr = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if frame is None:
            return {
                'status': 'error',
                'message': 'Could not decode image',
                'frames': [],
                'confidences': []
            }
            
        # Preprocess the frame
        img = preprocess_image(frame)
        if img is None:
            return {
                'status': 'error',
                'message': 'Error preprocessing image',
                'frames': [],
                'confidences': []
            }
            
        # Select appropriate model based on source type
        model = webcam_model if source_type == 'webcam' else file_upload_model
            
        # Inference
        results = model(img, verbose=False)
        
        # Process predictions
        detections = []
        confidence_threshold = 0.25
        
        # Process each detection
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get confidence and class
                conf = float(box.conf[0])
                
                # Only process detections with confidence above threshold
                if conf > confidence_threshold:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # convert tensor to list
                    
                    # Scale coordinates to original image size
                    orig_h, orig_w = frame.shape[:2]
                    scale_x = orig_w / 640  # 640 is the model input size
                    scale_y = orig_h / 640
                    
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    
                    # Add detection
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': 0,  # Single class for accident
                        'class_name': 'Accident'
                    })
                    logger.info(f"Added accident detection with confidence: {conf}")
        
        # Convert frame to base64 if accidents detected
        frames = []
        confidences = []
        if detections:
            # Draw boxes on frame copy
            frame_with_boxes = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_with_boxes, f"Accident {det['confidence']:.2f}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (0, 0, 255), 2)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Add frame and confidence
            frames.append(frame_base64)
            confidences.append(max(d['confidence'] for d in detections))
        
        logger.info(f"Returning {len(detections)} detections")
        return {
            'status': 'success',
            'message': 'Accident detected' if detections else 'No accident detected',
            'frames': frames,
            'confidences': confidences,
            'label': 'Accident' if detections else 'No Accident',
            'total_frames': 1,
            'accident_frames': len(detections),
            'duration': 0
        }
        
    except Exception as e:
        logger.error(f"Error in accident detection: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'message': f'Error in accident detection: {str(e)}',
            'frames': [],
            'confidences': []
        }

def detect_accidents_file_mode(frame, model):
    """Perform accident detection on a single frame using the file upload mode logic."""
    try:
        # Resize frame
        resize_width = 640
        resize_height = 640  # Changed to 640 to match YOLO's expected input size
        frame_resized = cv2.resize(frame, (resize_width, resize_height))
        
        logger.info("Running inference with file upload model")
        # Run inference
        results = model(frame_resized, verbose=False)
        
        detections = []
        # Process each detection
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # convert tensor to list
                conf = float(box.conf[0])
                
                # Only process detections with confidence above threshold
                if conf > 0.25:  # Confidence threshold
                    # Scale coordinates back to original size
                    scale_x = frame.shape[1] / resize_width
                    scale_y = frame.shape[0] / resize_height
                    
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': 0,  # Single class for accident
                        'class_name': 'Accident'
                    })
                    logger.info(f"Added accident detection with confidence: {conf}")
        
        logger.info(f"Returning {len(detections)} detections")
        return detections
        
    except Exception as e:
        logger.error(f"Error in file mode accident detection: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# Initialize Firebase (if credentials exist)
db = None
bucket = None
try:
    FIREBASE_CRED_FILE = "kapaan-19200503-firebase-adminsdk-fbsvc-8ca004f334.json"
    logger.info(f"Looking for Firebase credentials file at: {os.path.abspath(FIREBASE_CRED_FILE)}")
    
    if os.path.exists(FIREBASE_CRED_FILE):
        logger.info("Firebase credentials file found")
        try:
            # Read and validate the credentials file
            with open(FIREBASE_CRED_FILE, 'r') as f:
                cred_json = json.load(f)
                required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in cred_json]
                if missing_fields:
                    raise ValueError(f"Credentials file is missing required fields: {missing_fields}")
                logger.info("Credentials file is valid")
            
            # Use the existing Firebase initialization from firebase_config
            _, db, bucket = initialize_firebase()
            
            # Verify bucket exists and create directory structure
            try:
                bucket_name = f"{cred_json['project_id']}.firebasestorage.app"
                if not bucket.exists():
                    logger.error(f"Bucket gs://{bucket_name} does not exist")
                    logger.error("Please create the bucket in Firebase Console:")
                    logger.error("1. Go to: https://console.firebase.google.com/project/kaapan-14453/storage")
                    logger.error("2. Click 'Get Started' or 'Create bucket'")
                    logger.error("3. Select region: asia-south1 (Mumbai)")
                    logger.error("4. Start in test mode")
                    raise Exception("Storage bucket does not exist")
                
                logger.info(f"Successfully connected to Firebase Storage bucket: gs://{bucket_name}")
                
                # Create directory structure
                directories = [
                    'accident_videos/file/',
                    'accident_videos/webcam/'
                ]
                
                for directory in directories:
                    # Create a placeholder file to ensure directory exists
                    blob = bucket.blob(f"{directory}.keep")
                    if not blob.exists():
                        blob.upload_from_string(
                            '',
                            content_type='application/x-empty'
                        )
                        logger.info(f"Created directory: gs://{bucket_name}/{directory}")
                    else:
                        logger.info(f"Directory already exists: gs://{bucket_name}/{directory}")
                
                # Verify the structure by listing directories
                blobs = bucket.list_blobs(prefix='accident_videos/')
                existing_paths = [blob.name for blob in blobs]
                logger.info("Current storage structure:")
                logger.info(f"gs://{bucket_name}/")
                logger.info("└── accident_videos/")
                logger.info("    ├── file/")
                logger.info("    └── webcam/")
                
            except Exception as bucket_error:
                logger.error(f"Error accessing storage bucket: {str(bucket_error)}")
                bucket = None
                raise
                
            logger.info("Firebase Admin SDK initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase with credentials: {str(e)}")
            logger.error(traceback.format_exc())
            db = None
            bucket = None
            raise
    else:
        logger.error(f"Firebase credentials file not found at: {os.path.abspath(FIREBASE_CRED_FILE)}")
        db = None
        bucket = None
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    logger.error(traceback.format_exc())
    db = None
    bucket = None

def ensure_accidents_collection_exists():
    """Ensure the accidents collection exists before performing operations."""
    if not db:
        raise ValueError("Firestore client is not initialized")
        
    try:
        accidents_ref = db.collection('accidents')
        docs = accidents_ref.limit(1).stream()  # Use stream() instead of get()
        
        # Check if collection exists by trying to get first document
        try:
            next(docs)
            return True
        except StopIteration:
            # Collection is empty or doesn't exist, create it
            test_doc = {
                "id": "temp",
                "timestamp": datetime.now(),
                "test": True
            }
            temp_ref = accidents_ref.document("temp")
            temp_ref.set(test_doc)
            temp_ref.delete()
            return True
    except Exception as e:
        logger.error(f"Error ensuring accidents collection exists: {str(e)}")
        raise

def is_valid_base64(s: str) -> bool:
    """Check if a string is valid base64."""
    try:
        # Check if string is properly padded
        if len(s) % 4:
            # Add padding if necessary
            s += '=' * (4 - len(s) % 4)
        # Try to decode
        base64.b64decode(s)
        return True
    except Exception:
        return False

def sanitize_base64_image(image_data: str) -> str:
    """Sanitize and validate base64 image data."""
    try:
        # If the string starts with data URI scheme, extract the base64 part
        if image_data.startswith('data:image'):
            # Extract the base64 part after the comma
            image_data = image_data.split(',', 1)[1]
        
        # Remove any whitespace
        image_data = image_data.strip()
        
        # Add padding if necessary
        if len(image_data) % 4:
            image_data += '=' * (4 - len(image_data) % 4)
            
        # Validate the base64 string
        if not is_valid_base64(image_data):
            logger.warning("Invalid base64 image data provided")
            return ""
            
        return image_data
    except Exception as e:
        logger.error(f"Error sanitizing base64 image: {str(e)}")
        return ""

def upload_video_to_storage(video_data: bytes, doc_id: str, source_type: str, detected_frames: list = None, file_extension: str = 'mp4') -> dict:
    """
    Upload video and detected frames to Firebase Storage
    """
    if not storage_bucket:
        raise ValueError("Firebase Storage is not initialized")
    
    try:
        urls = {}
        
        # Upload video
        video_path = f"accident_videos/{doc_id}/video.{file_extension}"
        video_blob = storage_bucket.blob(video_path)
        video_blob.upload_from_string(video_data, content_type=f'video/{file_extension}')
        video_blob.make_public()
        urls['video_url'] = video_blob.public_url
        logger.info(f"Video uploaded successfully to: {urls['video_url']}")
        
        # Upload detected frames if available
        if detected_frames:
            frame_urls = []
            for idx, frame in enumerate(detected_frames[:3]):  # Limit to top 3 frames
                try:
                    # Get frame data
                    frame_data = frame['image_data']
                    
                    # Upload frame
                    frame_path = f"accident_videos/{doc_id}/frames/frame_{idx + 1}.jpg"
                    frame_blob = storage_bucket.blob(frame_path)
                    frame_blob.upload_from_string(frame_data, content_type='image/jpeg')
                    frame_blob.make_public()
                    
                    frame_urls.append({
                        'url': frame_blob.public_url,
                        'confidence': frame['confidence'],
                        'timestamp': frame['timestamp']
                    })
                    logger.info(f"Frame {idx + 1} uploaded successfully to: {frame_blob.public_url}")
                except Exception as e:
                    logger.error(f"Error uploading frame {idx + 1}: {str(e)}")
                    continue
            
            urls['frame_urls'] = frame_urls
            logger.info(f"Successfully uploaded {len(frame_urls)} frames")
        
        return urls
        
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise

def create_accident_document(data: dict) -> str:
    """Create a new accident document in Firestore"""
    if not firestore_db:
        raise ValueError("Firestore client is not initialized")
    
    try:
        # Process the data
        doc_id = data['id']
        
        # Update detected frames with storage URLs if available
        if 'frame_urls' in data.get('video_data', {}):
            detected_frames = []
            for frame_info in data['video_data']['frame_urls']:
                detected_frames.append({
                    'image_url': frame_info['url'].strip(),  # Ensure clean URL
                    'confidence': frame_info['confidence'],
                    'timestamp': frame_info['timestamp']
                })
            data['detected_frames'] = detected_frames
        
        # Add the document to Firestore
        doc_ref = firestore_db.collection('accidents').document(doc_id)
        doc_ref.set(data)
        logger.info(f"Created accident document with ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Error creating accident document: {str(e)}")
        raise

async def get_accident_by_id(doc_id: str) -> dict:
    """Get an accident document by ID."""
    try:
        # Ensure collection exists
        ensure_accidents_collection_exists()
        
        doc = await db.collection('accidents').document(doc_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error(f"Error getting accident document: {str(e)}")
        raise

async def get_recent_accidents(limit: int = 10) -> list:
    """Get most recent accidents."""
    try:
        # Ensure collection exists
        ensure_accidents_collection_exists()
        
        docs = await db.collection('accidents')\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(limit)\
            .get()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        logger.error(f"Error getting recent accidents: {str(e)}")
        raise

async def get_accidents_by_location(lat: float, lng: float, radius_km: float = 1.0) -> list:
    """Get accidents within radius of a location."""
    try:
        # Ensure collection exists
        ensure_accidents_collection_exists()
        
        # Convert km to lat/lng degrees (approximate)
        lat_degree = radius_km / 111.32  # 1 degree = 111.32 km
        lng_degree = radius_km / (111.32 * math.cos(math.radians(lat)))
        
        # Get bounds
        lat_min = lat - lat_degree
        lat_max = lat + lat_degree
        lng_min = lng - lng_degree
        lng_max = lng + lng_degree
        
        # Query using bounds
        docs = await db.collection('accidents')\
            .where('location', '>', firestore.GeoPoint(lat_min, lng_min))\
            .where('location', '<', firestore.GeoPoint(lat_max, lng_max))\
            .get()
            
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        logger.error(f"Error getting accidents by location: {str(e)}")
        raise

# Global flag to track processing state
is_processing = True

@app.post("/stop_processing")
async def stop_processing():
    global is_processing
    is_processing = False
    logger.info("Processing stopped by client")
    return {"status": "stopped"}

def compress_image_base64(base64_string, max_size_kb=200):
    """Compress a base64 image to ensure it's under the specified size."""
    try:
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        # Initial quality
        quality = 95
        output = io.BytesIO()
        
        while True:
            # Clear the output buffer
            output.seek(0)
            output.truncate(0)
            
            # Save with current quality
            img.save(output, format='JPEG', quality=quality)
            size_kb = len(output.getvalue()) / 1024
            
            # If size is under max_size_kb or quality is minimum, return the result
            if size_kb <= max_size_kb or quality <= 20:
                output.seek(0)
                return base64.b64encode(output.getvalue()).decode('utf-8')
            
            # Reduce quality for next iteration
            quality -= 5

    except Exception as e:
        logger.error(f"Error compressing image: {str(e)}")
        return base64_string  # Return original if compression fails

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def process_video(video_path):
    """Process a video file for accident detection."""
    try:
        # Load the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'status': 'error',
                'message': 'Could not open video file',
                'frames': [],
                'confidences': []
            }

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Initialize variables
        frames = []
        confidences = []
        frame_count = 0
        accident_frames = 0
        skip_frames = 5  # Process every 5th frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Run inference
            detections = detect_accidents_file_mode(frame, file_upload_model)
            
            # If accidents detected, store the frame
            if detections:
                accident_frames += 1
                
                # Draw boxes on frame
                frame_with_boxes = frame.copy()
                for det in detections:
                    try:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        conf = det['confidence']
                        
                        # Draw rectangle and confidence
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame_with_boxes, 
                                  f"Accident {conf:.2f}", 
                                  (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9,
                                  (0, 0, 255),
                                  2)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.error(f"Error drawing detection box: {str(e)}")
                        continue
                
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', frame_with_boxes)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Store frame and highest confidence
                frames.append(frame_base64)
                confidences.append(max(d['confidence'] for d in detections))

        # Release video capture
        cap.release()
        
        # Calculate detection percentage
        processed_frames = total_frames // skip_frames
        detection_percentage = (accident_frames / max(1, processed_frames)) * 100

        return {
            'status': 'success',
            'message': 'Video processed successfully',
            'frames': frames,
            'confidences': confidences,
            'label': 'Accident' if accident_frames > 0 else 'No Accident',
            'total_frames': total_frames,
            'accident_frames': accident_frames,
            'duration': duration,
            'detection_percentage': detection_percentage
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'message': f'Error processing video: {str(e)}',
            'frames': [],
            'confidences': []
        }

def convert_frame_to_base64(frame: np.ndarray) -> str:
    """Convert a frame to base64 string."""
    try:
        # Ensure frame is in BGR format
        if len(frame.shape) == 2:  # If grayscale, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Encode frame to jpg format
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert to base64 string
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str
    except Exception as e:
        logger.error(f"Error converting frame to base64: {str(e)}")
        return ""

@app.post("/predict")
async def predict(file: UploadFile = File(...), source_type: str = Form(...), accident_data: str = Form(...)):
    """
    Process uploaded video file or webcam frame and detect accidents.
    For webcam mode, it processes individual frames and maintains best frames.
    """
    try:
        if source_type == "webcam":
            # Read the frame data
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame
            result = process_webcam_frame(frame, webcam_model)
            if not result:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Error processing webcam frame"}
                )
            
            # Convert frame to base64 for response
            _, buffer = cv2.imencode('.jpg', result['frame_with_boxes'])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "label": "Accident" if result['is_accident'] else "No Accident",
                    "confidence": float(result['confidence']),
                    "frame": frame_base64,
                    "timestamp": result['timestamp']
                }
            )
            
        else:  # File upload mode
            temp_dir = None
            try:
                # Parse accident data
                accident_data_dict = json.loads(accident_data)
                location_data = accident_data_dict.get('location', {})
                latitude = location_data.get('latitude', DEFAULT_LATITUDE)
                longitude = location_data.get('longitude', DEFAULT_LONGITUDE)

                # Create temporary directory for frames
                temp_dir = tempfile.mkdtemp()
                
                # Generate unique document ID
                doc_id = str(uuid.uuid4())
                logger.info(f"Generated document ID: {doc_id}")

                # Update status: Uploading
                await update_status("Uploading video file...")

                # Save uploaded file temporarily
                temp_file_path = os.path.join(temp_dir, "temp_video.mp4")
                with open(temp_file_path, "wb") as f:
                    f.write(await file.read())
                logger.info("Video file received and saved temporarily")

                # Open video capture
                cap = cv2.VideoCapture(temp_file_path)
                if not cap.isOpened():
                    return JSONResponse(content={"error": "Error opening video"})

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                frame_skip = max(1, int(fps / 4))  # Process 4 frames per second
                logger.info("Starting accident detection...")
                
                # Update status: Starting detection
                await update_status("Starting accident detection...")
                
                # Initialize variables
                processed_frames = 0
                accident_frames = []
                accident_confidences = []
                accident_detected = False
                confidence_threshold = 0.25  # Lowered threshold to match your example

                # Process video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Only process every nth frame
                    if processed_frames % frame_skip != 0:
                        processed_frames += 1
                        continue

                    # Run accident detection
                    detections = detect_accidents_file_mode(frame, file_upload_model)
                    
                    # Check for accidents in this frame
                    frame_confidence = 0
                    if detections:
                        # Get highest confidence detection for this frame
                        frame_confidence = max(det['confidence'] for det in detections)
                        
                        if frame_confidence > confidence_threshold:
                            # Draw bounding boxes on frame
                            frame_with_boxes = frame.copy()
                            for det in detections:
                                x1, y1, x2, y2 = map(int, det['bbox'])
                                conf = det['confidence']
                                
                                # Draw detection box
                                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame_with_boxes, 
                                          f"Accident {conf:.2f}", 
                                          (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.9,
                                          (0, 0, 255),
                                          2)
                        
                            # Save frame as image file
                            frame_path = os.path.join(temp_dir, f"frame_{len(accident_frames)}.jpg")
                            cv2.imwrite(frame_path, frame_with_boxes)
                            
                            # Store frame path and confidence
                            accident_frames.append(frame_path)
                            accident_confidences.append(frame_confidence)
                            accident_detected = True
                            logger.info(f"Detected accident with confidence: {frame_confidence}")

                    processed_frames += 1

                # Release resources
                cap.release()
                logger.info("Accident detection completed")
                
                # Upload video and frames if accidents detected
                if accident_detected and len(accident_frames) > 0:
                    try:
                        logger.info(f"Processing {len(accident_frames)} accident frames for storage")
                        
                        # Take top 3 frames by confidence
                        frame_data = list(zip(accident_frames, accident_confidences))
                        frame_data.sort(key=lambda x: x[1], reverse=True)
                        top_frames = frame_data[:3]

                        # Prepare frames for storage
                        detected_frames_data = []
                        base_time = datetime.now()
                        logger.info("Preparing frames for upload...")
                        
                        # Read frames and prepare for upload
                        for idx, (frame_path, conf) in enumerate(top_frames):
                            try:
                                with open(frame_path, 'rb') as f:
                                    frame_data = f.read()
                                detected_frames_data.append({
                                    'image_data': frame_data,
                                    'confidence': conf,
                                    'timestamp': (base_time + timedelta(seconds=idx)).isoformat()
                                })
                                logger.info(f"Prepared frame {idx + 1} with confidence {conf}")
                            except Exception as e:
                                logger.error(f"Error reading frame {idx + 1}: {str(e)}")
                                continue

                        # Read video file for upload
                        with open(temp_file_path, 'rb') as f:
                            video_data = f.read()

                        # Update status: Uploading to storage
                        await update_status("Uploading to Firebase Storage...")
                        
                        # Upload to Firebase Storage
                        logger.info("Uploading files to Firebase Storage...")
                        storage_urls = upload_video_to_storage(
                            video_data,
                            doc_id,
                            'file',
                            detected_frames_data,
                            'mp4'
                        )
                        logger.info("Upload to Firebase Storage complete")

                        if not storage_urls.get('frame_urls'):
                            raise ValueError("No frame URLs returned from storage upload")

                        # Update status: Creating document
                        await update_status("Creating Firestore document...")

                        # Create Firestore document
                        logger.info("Creating Firestore document...")
                        accident_doc = {
                            'id': doc_id,
                            'detected_frames': [
                                {
                                    'confidence': frame_info['confidence'],
                                    'image_url': frame_info['url'].strip(),  # Ensure clean URL
                                    'timestamp': frame_info['timestamp']
                                }
                                for frame_info in storage_urls.get('frame_urls', [])
                            ],
                            'location': {
                                'latitude': latitude,
                                'longitude': longitude
                            },
                            'metadata': {
                                'detection_model': 'uyir.pt',
                                'processed_at': datetime.utcnow().isoformat() + 'Z',
                                'source_type': 'file',
                                'timestamp': datetime.utcnow().isoformat() + 'Z'
                            },
                            'video_data': {
                                'accident_frames': len(accident_frames),
                                'duration': duration,
                                'format': 'mp4',
                                'frame_urls': [
                                    {
                                        'confidence': frame_info['confidence'],
                                        'timestamp': frame_info['timestamp'],
                                        'url': frame_info['url'].strip()  # Ensure clean URL
                                    }
                                    for frame_info in storage_urls.get('frame_urls', [])
                                ],
                                'total_frames': total_frames,
                                'video_url': storage_urls.get('video_url', '').strip()  # Ensure clean URL
                            }
                        }

                        # Store in Firestore
                        doc_id = create_accident_document(accident_doc)
                        logger.info(f"Created accident document with ID: {doc_id}")

                        # Update status: Accident detected
                        await update_status("Accident detected! Displaying results...")

                        # Return response in format expected by frontend
                        frames_for_frontend = [
                            frame['image_url'] for frame in accident_doc['detected_frames']
                        ]
                        
                        return JSONResponse(content={
                            'status': 'success',
                            'label': "Accident Detected",
                            'frames': frames_for_frontend,
                            'average_detection_percentage': round((len(accident_frames) / processed_frames) * 100, 2),
                            'doc_id': doc_id,
                            'video_url': storage_urls.get('video_url', '').strip(),
                            'accident_data': accident_doc,
                            'progress': 'complete'
                        })

                    except Exception as e:
                        logger.error(f"Error storing accident data: {str(e)}")
                        logger.error(traceback.format_exc())
                        await update_status(f"Error: {str(e)}")
                        return JSONResponse(content={
                            'error': 'Error storing accident data',
                            'message': str(e),
                            'progress': 'error'
                        })

                else:
                    await update_status("No accidents detected")
                    return JSONResponse(content={
                        'status': 'success',
                        'label': "No Accident",
                        'frames': [],
                        'average_detection_percentage': 0,
                        'total_frames': total_frames,
                        'accident_frames': 0,
                        'progress': 'complete'
                    })

            except Exception as e:
                logger.error(f"Error in predict endpoint: {str(e)}")
                logger.error(traceback.format_exc())
                await update_status(f"Error: {str(e)}")
                return JSONResponse(content={
                    'error': 'Internal Server Error',
                    'message': str(e),
                    'progress': 'error'
                })
            finally:
                # Clean up temp files
                try:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Could not remove temporary directory: {str(e)}")

    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        await update_status(f"Error: {str(e)}")
        return JSONResponse(content={
            'error': 'Internal Server Error',
            'message': str(e),
            'progress': 'error'
        })

def get_ffmpeg_path():
    """Get the path to FFmpeg executable."""
    # First, check if ffmpeg is in the current directory
    local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'ffmpeg')
    if os.path.exists(local_ffmpeg):
        logger.info(f"Using local FFmpeg: {local_ffmpeg}")
        return local_ffmpeg
    
    # Then, check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        logger.info(f"Using system FFmpeg: {ffmpeg_path}")
        return ffmpeg_path
    
    logger.error("FFmpeg not found. Please ensure ffmpeg is in the project directory or system PATH")
    return None

def convert_video_to_mp4(input_path):
    """Convert video to MP4 format using ffmpeg."""
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        logger.error("FFmpeg not found. Please install FFmpeg or place ffmpeg in the project directory.")
        return None

    output_path = input_path + '.mp4'
    try:
        # Use ffmpeg to convert the video with more robust settings
        command = [
            ffmpeg_path,
            '-i', input_path,  # Input file
            '-c:v', 'libx264',  # Video codec
            '-preset', 'ultrafast',  # Encoding preset
            '-pix_fmt', 'yuv420p',  # Pixel format
            '-movflags', '+faststart',  # Enable fast start
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        # Run the command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg conversion error: {stderr.decode()}")
            return None
            
        return output_path
    except Exception as e:
        logger.error(f"Video conversion error: {str(e)}")
        return None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
        # Replace localhost with actual IP address for both HTTP and HTTPS
        html_content = html_content.replace(
            'http://127.0.0.1:8000',
            f'http://{LOCAL_IP}:8080'  # HTTP port
        ).replace(
            'https://127.0.0.1:8000',
            f'https://{LOCAL_IP}:8443'  # HTTPS port
        )
        return html_content

def extract_accident_clip(video_data, frame_timestamps, margin_seconds=2):
    """
    Extract a video clip containing the accident frames with margin before and after.
    
    Args:
        video_data: bytes containing the video data
        frame_timestamps: list of timestamps in milliseconds
        margin_seconds: seconds of margin to include before and after each accident frame
    
    Returns:
        tuple: (video_clip_bytes, clip_duration)
    """
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save input video
            input_path = os.path.join(temp_dir, 'input.webm')  # Changed to .webm for webcam
            with open(input_path, 'wb') as f:
                f.write(video_data)
            
            # Get video info using ffmpeg
            probe = ffmpeg.probe(input_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Try to get duration from different sources
            duration = None
            try:
                duration = float(probe['format']['duration'])
            except (KeyError, ValueError):
                try:
                    # Try to get duration from video stream
                    duration = float(video_info.get('duration', 0))
                except (KeyError, ValueError):
                    # If no duration found, estimate from timestamps
                    if frame_timestamps:
                        duration = max(frame_timestamps) / 1000.0 + margin_seconds
                    else:
                        duration = 30.0  # Default 30 seconds if no information available
            
            # Convert timestamps to seconds
            timestamps_sec = sorted([ts / 1000.0 for ts in frame_timestamps])  # Convert ms to seconds
            
            # Create segments around each accident frame
            segments = []
            for ts in timestamps_sec:
                start = max(0, ts - margin_seconds)
                end = min(duration, ts + margin_seconds)
                segments.append((start, end))
            
            # Merge overlapping segments
            merged_segments = []
            if segments:
                current_start, current_end = segments[0]
                for start, end in segments[1:]:
                    if start <= current_end:
                        current_end = max(current_end, end)
                    else:
                        merged_segments.append((current_start, current_end))
                        current_start, current_end = start, end
                merged_segments.append((current_start, current_end))
            
            if not merged_segments:
                logger.warning("No valid segments found")
                return None, 0
            
            # Create segment files
            segment_files = []
            for i, (start, end) in enumerate(merged_segments):
                output_segment = os.path.join(temp_dir, f'segment_{i}.mp4')
                try:
                    # Cut segment using ffmpeg
                    stream = ffmpeg.input(input_path, ss=start, t=end-start)
                    stream = ffmpeg.output(stream, output_segment, 
                                        acodec='aac', 
                                        vcodec='libx264',
                                        preset='ultrafast',
                                        video_bitrate='2500k')
                    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    segment_files.append(output_segment)
                except ffmpeg.Error as e:
                    logger.error(f"FFmpeg error: {e.stderr.decode()}")
                    continue
            
            if not segment_files:
                logger.warning("No segments were successfully created")
                return None, 0
            
            # If only one segment, return it directly
            if len(segment_files) == 1:
                with open(segment_files[0], 'rb') as f:
                    clip_data = f.read()
                    # Get duration of the clip
                    try:
                        probe = ffmpeg.probe(segment_files[0])
                        clip_duration = float(probe['format']['duration'])
                    except (KeyError, ValueError, ffmpeg.Error):
                        # If we can't get the duration, estimate it
                        clip_duration = end - start
                    return clip_data, clip_duration
            
            # Concatenate segments
            concat_file = os.path.join(temp_dir, 'concat.txt')
            with open(concat_file, 'w') as f:
                for segment in segment_files:
                    f.write(f"file '{segment}'\n")
            
            final_output = os.path.join(temp_dir, 'final_output.mp4')
            try:
                # Concatenate using ffmpeg
                stream = ffmpeg.input(concat_file, f='concat', safe=0)
                stream = ffmpeg.output(stream, final_output, c='copy')
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                
                # Get final duration
                try:
                    probe = ffmpeg.probe(final_output)
                    final_duration = float(probe['format']['duration'])
                except (KeyError, ValueError, ffmpeg.Error):
                    # If we can't get the duration, estimate it from segments
                    final_duration = sum(end - start for start, end in merged_segments)
                
                # Read final output
                with open(final_output, 'rb') as f:
                    return f.read(), final_duration
                
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg concatenation error: {e.stderr.decode()}")
                # If concatenation fails, return the first segment
                with open(segment_files[0], 'rb') as f:
                    clip_data = f.read()
                    try:
                        probe = ffmpeg.probe(segment_files[0])
                        clip_duration = float(probe['format']['duration'])
                    except (KeyError, ValueError, ffmpeg.Error):
                        # If we can't get the duration, estimate it
                        clip_duration = merged_segments[0][1] - merged_segments[0][0]
                    return clip_data, clip_duration
            
    except Exception as e:
        logger.error(f"Error extracting accident clip: {str(e)}")
        logger.error(traceback.format_exc())
        return None, 0

@app.get("/sample_accidents")
async def get_sample_accidents():
    """Get sample accident data for testing."""
    try:
        # Create sample accident data
        current_time = datetime.now()
        sample_data = [
            {
                "id": str(uuid.uuid4()),
                "accident_detected": True,
                "average_detection_percentage": 85.5,
                "detected_frames": [
                    {
                        "image": "base64_string_frame_1",  # Replace with actual base64 image
                        "confidence": 0.95,
                        "timestamp": current_time.isoformat()
                    },
                    {
                        "image": "base64_string_frame_2",  # Replace with actual base64 image
                        "confidence": 0.88,
                        "timestamp": (current_time + timedelta(seconds=1)).isoformat()
                    },
                    {
                        "image": "base64_string_frame_3",  # Replace with actual base64 image
                        "confidence": 0.82,
                        "timestamp": (current_time + timedelta(seconds=2)).isoformat()
                    }
                ],
                "video_data": {
                    "video": "",  # base64 string for video
                    "format": "mp4",
                    "duration": 15.5,
                    "total_frames": 450,
                    "accident_frames": 35
                },
                "intensity": "High",
                "location": {
                    "latitude": DEFAULT_LATITUDE,
                    "longitude": DEFAULT_LONGITUDE
                },
                "persons_involved": 2,
                "reported": True,
                "timestamp": current_time.isoformat(),
                "metadata": {
                    "source_type": "test",
                    "processed_at": current_time.isoformat(),
                    "detection_model": "YOLOv8",
                    "frame_timestamps": [
                        current_time.isoformat(),
                        (current_time + timedelta(seconds=1)).isoformat(),
                        (current_time + timedelta(seconds=2)).isoformat()
                    ]
                }
            },
            {
                "id": str(uuid.uuid4()),
                "accident_detected": True,
                "average_detection_percentage": 92.3,
                "detected_frames": [
                    {
                        "image": "base64_string_frame_4",  # Replace with actual base64 image
                        "confidence": 0.97,
                        "timestamp": current_time.isoformat()
                    },
                    {
                        "image": "base64_string_frame_5",  # Replace with actual base64 image
                        "confidence": 0.94,
                        "timestamp": (current_time + timedelta(seconds=1)).isoformat()
                    },
                    {
                        "image": "base64_string_frame_6",  # Replace with actual base64 image
                        "confidence": 0.91,
                        "timestamp": (current_time + timedelta(seconds=2)).isoformat()
                    }
                ],
                "video_data": {
                    "video": "",  # base64 string for video
                    "format": "mp4",
                    "duration": 20.0,
                    "total_frames": 600,
                    "accident_frames": 45
                },
                "intensity": "Medium",
                "location": {
                    "latitude": DEFAULT_LATITUDE + 0.01,
                    "longitude": DEFAULT_LONGITUDE + 0.01
                },
                "persons_involved": 3,
                "reported": True,
                "timestamp": (current_time - timedelta(hours=1)).isoformat(),
                "metadata": {
                    "source_type": "webcam",
                    "processed_at": (current_time - timedelta(hours=1)).isoformat(),
                    "detection_model": "YOLOv8",
                    "frame_timestamps": [
                        (current_time - timedelta(hours=1)).isoformat(),
                        (current_time - timedelta(hours=1) + timedelta(seconds=1)).isoformat(),
                        (current_time - timedelta(hours=1) + timedelta(seconds=2)).isoformat()
                    ]
                }
            }
        ]
        
        return JSONResponse(status_code=200, content={"accidents": sample_data})
        
    except Exception as e:
        logger.error(f"Error getting sample accidents: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting sample accidents: {str(e)}"}
        )

@app.get("/status")
async def status_updates():
    """
    Server-Sent Events (SSE) endpoint for status updates.
    """
    async def event_generator():
        try:
            # Initial connection established
            yield "data: {\"status\": \"Connected to status updates\"}\n\n"
            
            while not shutdown_event.is_set():
                try:
                    # Wait for status updates with a timeout
                    status = await asyncio.wait_for(status_updates_queue.get(), timeout=1.0)
                    yield f"data: {{\"status\": \"{status}\"}}\n\n"
                except asyncio.TimeoutError:
                    # Send keep-alive comment
                    yield ": keep-alive\n\n"
                except Exception as e:
                    logger.error(f"Error in status updates: {str(e)}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("Client disconnected from status updates")
        except Exception as e:
            logger.error(f"Error in event generator: {str(e)}")
        finally:
            logger.info("Status updates connection closed")
    
    return EventSourceResponse(event_generator())

async def update_status(status: str):
    """Helper function to send status updates"""
    try:
        await status_updates_queue.put(status)
    except Exception as e:
        logger.error(f"Error updating status: {str(e)}")

def is_overlap(boxA, boxB):
    """Check if two bounding boxes overlap."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

def detect_accidents_webcam_mode(frame, model):
    """
    Detect accidents in webcam mode using bounding box intersection.
    Returns:
    - is_accident: bool
    - confidence: float
    - boxes: list of collided boxes
    - class_names: list of class names
    """
    # Define vehicle classes (COCO)
    vehicle_classes = ['car', 'bus', 'motorbike', 'truck']
    
    # Run inference
    results = model(frame, verbose=False)[0]
    vehicle_boxes = []
    vehicle_info = []  # Store box coordinates and class names

    if results.boxes is not None:
        # Collect all vehicle boxes
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            class_name = model.names[int(cls_id)]
            if class_name in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                vehicle_boxes.append([x1, y1, x2, y2])
                vehicle_info.append({
                    'box': [x1, y1, x2, y2],
                    'class': class_name
                })

        # Find collisions
        collided_indices = set()
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                if is_overlap(vehicle_boxes[i], vehicle_boxes[j]):
                    collided_indices.add(i)
                    collided_indices.add(j)

        # If collision detected, prepare return data
        if collided_indices:
            # Get collided boxes and their classes
            collided_boxes = [vehicle_info[i]['box'] for i in collided_indices]
            collided_classes = [vehicle_info[i]['class'] for i in collided_indices]
            
            # Calculate overlap area and confidence for each pair
            pair_confidences = []
            for i in collided_indices:
                for j in collided_indices:
                    if i < j:
                        box1 = vehicle_boxes[i]
                        box2 = vehicle_boxes[j]
                        # Calculate intersection area
                        xA = max(box1[0], box2[0])
                        yA = max(box1[1], box2[1])
                        xB = min(box1[2], box2[2])
                        yB = min(box1[3], box2[3])
                        if xA < xB and yA < yB:
                            # Calculate intersection area
                            intersection_area = (xB - xA) * (yB - yA)
                            # Calculate union area
                            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            union_area = box1_area + box2_area - intersection_area
                            # Calculate IoU (Intersection over Union)
                            iou = intersection_area / union_area if union_area > 0 else 0
                            # Add to confidences
                            pair_confidences.append(iou)
            
            # Calculate final confidence as average of pair IoUs
            confidence = sum(pair_confidences) / len(pair_confidences) if pair_confidences else 0
            # Scale confidence to be more conservative
            confidence = confidence * 0.8  # Scale down to avoid over-confidence
            
            return True, confidence, collided_boxes, collided_classes

    return False, 0.0, [], []

def process_webcam_frame(frame, model):
    """
    Process a single webcam frame and return detection results.
    """
    try:
        # Convert frame to RGB if needed
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # If RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Detect accidents
        is_accident, confidence, boxes, classes = detect_accidents_webcam_mode(frame, model)
        
        # Draw boxes and text on frame
        frame_with_boxes = frame.copy()
        if is_accident:
            # Draw connecting lines between colliding vehicles
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    # Get centers of boxes
                    center1 = ((boxes[i][0] + boxes[i][2]) // 2, (boxes[i][1] + boxes[i][3]) // 2)
                    center2 = ((boxes[j][0] + boxes[j][2]) // 2, (boxes[j][1] + boxes[j][3]) // 2)
                    # Draw red line connecting centers
                    cv2.line(frame_with_boxes, center1, center2, (0, 0, 255), 2)

            # Draw boxes for each vehicle
            for box, class_name in zip(boxes, classes):
                x1, y1, x2, y2 = box
                # Draw red box
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Draw vehicle class and warning
                text = f"{class_name} - COLLISION!"
                cv2.putText(frame_with_boxes, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw overall accident confidence
            cv2.putText(frame_with_boxes, f"ACCIDENT DETECTED - Overlap: {confidence:.2%}", (30, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_with_boxes, "No Collision Detected", (30, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return {
            'is_accident': is_accident,
            'confidence': confidence,
            'boxes': boxes,
            'classes': classes,
            'frame': frame,
            'frame_with_boxes': frame_with_boxes,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing webcam frame: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def upload_webcam_video_to_storage(video_bytes: bytes, doc_id: str) -> dict:
    """
    Upload webcam video to Firebase Storage.
    Returns video URL if successful.
    """
    try:
        bucket = storage.bucket()
        video_path = f"accident_videos/{doc_id}/video.webm"
        blob = bucket.blob(video_path)
        
        # Upload video
        blob.upload_from_string(
            video_bytes,
            content_type='video/webm'
        )
        
        # Make public and get URL
        blob.make_public()
        video_url = blob.public_url
        
        logger.info(f"Video uploaded successfully to: {video_url}")
        return {
            'video_url': video_url
        }
        
    except Exception as e:
        logger.error(f"Error uploading webcam video: {str(e)}")
        return None

async def upload_webcam_frames_to_storage(frame_data_list: list, doc_id: str) -> list:
    """
    Upload webcam frames to Firebase Storage.
    Returns list of frame URLs and metadata.
    """
    try:
        bucket = storage.bucket()
        frame_urls = []
        
        for idx, frame_data in enumerate(frame_data_list):
            try:
                # Upload frame
                frame_path = f"accident_videos/{doc_id}/frames/frame_{idx + 1}.jpg"
                blob = bucket.blob(frame_path)
                
                # Decode base64 and upload
                image_data = base64.b64decode(frame_data['frame'])
                blob.upload_from_string(
                    image_data,
                    content_type='image/jpeg'
                )
                
                # Make public and get URL
                blob.make_public()
                frame_url = blob.public_url
                
                # Store frame info
                frame_urls.append({
                    'url': frame_url,
                    'confidence': frame_data['confidence'],
                    'timestamp': frame_data['timestamp']
                })
                
                logger.info(f"Successfully uploaded frame {idx + 1}")
                
            except Exception as e:
                logger.error(f"Error uploading frame {idx + 1}: {str(e)}")
        
        return frame_urls
        
    except Exception as e:
        logger.error(f"Error in upload_webcam_frames_to_storage: {str(e)}")
        return []

@app.post("/stop_webcam")
async def stop_webcam(
    video_data: UploadFile = File(...),
    best_frames: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """
    Handle webcam stop event.
    Upload collected data to Firebase if accidents were detected.
    """
    try:
        # Parse best frames data
        best_frames_data = json.loads(best_frames)
        
        # Only proceed if we have accident frames
        if not best_frames_data:
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": "No accidents detected"}
            )
        
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Save video to temporary file to get duration
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, "temp_video.webm")
        video_bytes = await video_data.read()
        
        try:
            # Save video temporarily
            with open(temp_video_path, "wb") as f:
                f.write(video_bytes)
            
            # Get video duration using ffmpeg
            probe = ffmpeg.probe(temp_video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            total_frames = int(float(video_info['nb_frames']) if 'nb_frames' in video_info else duration * float(video_info['r_frame_rate'].split('/')[0]))
            
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            duration = len(best_frames_data) / 30  # Fallback to estimation
            total_frames = len(best_frames_data)
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {str(e)}")
        
        # Upload video
        video_result = await upload_webcam_video_to_storage(video_bytes, doc_id)
        
        if not video_result:
            raise Exception("Failed to upload video")
        
        # Upload frames
        frame_urls = await upload_webcam_frames_to_storage(best_frames_data, doc_id)
        
        if not frame_urls:
            raise Exception("Failed to upload frames")
        
        # Create accident document data
        accident_data = {
            'id': doc_id,
            'accident_detected': True,
            'detected_frames': [
                {
                    'image_url': frame['url'],
                    'confidence': frame['confidence'],
                    'timestamp': frame['timestamp']
                } for frame in frame_urls
            ],
            'video_data': {
                'format': 'webm',
                'duration': duration,  # Use actual duration
                'total_frames': total_frames,  # Use actual frame count
                'accident_frames': len(best_frames_data),
                'video_url': video_result['video_url'],
                'frame_urls': frame_urls
            },
            'location': {
                'latitude': float(latitude),
                'longitude': float(longitude)
            },
            'metadata': {
                'source_type': 'webcam',
                'processed_at': datetime.now().isoformat(),
                'detection_model': 'YOLOv8',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Create Firestore document
        db = firestore.client()
        doc_ref = db.collection('accidents').document(doc_id)
        doc_ref.set(accident_data)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Data uploaded successfully",
                "doc_id": doc_id,
                "accident_data": accident_data
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stop_webcam endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

if __name__ == "__main__":
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8080,
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())