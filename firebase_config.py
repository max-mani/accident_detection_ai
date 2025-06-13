import os
from firebase_admin import credentials, initialize_app, storage, firestore, get_app
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to your Firebase credentials JSON file
FIREBASE_CREDENTIALS_PATH = "kapaan-19200503-firebase-adminsdk-fbsvc-8ca004f334.json"
STORAGE_BUCKET = "kapaan-19200503.firebasestorage.app"

def initialize_firebase():
    """Initialize Firebase with credentials"""
    try:
        # Try to get existing app
        app = get_app()
        db = firestore.client()
        bucket = storage.bucket()
        logger.info("Using existing Firebase app")
        return app, db, bucket
    except ValueError:
        # No app exists, initialize new one
        if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
            raise FileNotFoundError(
                f"Firebase credentials file not found at {FIREBASE_CREDENTIALS_PATH}. "
                "Please place your Firebase credentials JSON file in the project root."
            )
        
        try:
            # Initialize Firebase app with credentials
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            app = initialize_app(cred, {
                'storageBucket': STORAGE_BUCKET,
                'projectId': 'kapaan-19200503',
                'locationId': 'asia-south1'  # Mumbai region for better performance in Tamil Nadu
            })
            
            # Initialize Firestore client
            db = firestore.client()
            
            # Initialize Storage bucket
            bucket = storage.bucket()
            
            # Test storage bucket connection
            if not bucket.exists():
                logger.warning(f"Storage bucket {STORAGE_BUCKET} does not exist")
                logger.warning("Please create the bucket in Firebase Console:")
                logger.warning("1. Go to Firebase Console > Storage")
                logger.warning("2. Click 'Get Started'")
                logger.warning("3. Choose region: asia-south1 (Mumbai)")
                logger.warning("4. Set up security rules")
            else:
                logger.info(f"Successfully connected to storage bucket: {STORAGE_BUCKET}")
                
            # Test Firestore connection
            try:
                # Try to access a collection to verify Firestore connection
                db.collection('accidents').limit(1).get()
                logger.info("Successfully connected to Firestore")
            except Exception as e:
                logger.warning(f"Error testing Firestore connection: {str(e)}")
                logger.warning("Please ensure Firestore is enabled in Firebase Console")
            
            logger.info("Firebase app initialized successfully")
            return app, db, bucket
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            logger.error(traceback.format_exc())
            raise 