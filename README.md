# Accident Detection AI

## Overview
This project is an AI-powered accident detection system designed to analyze video streams and detect accidents in real-time. It leverages deep learning models and integrates with cloud services for alerting and data storage.

## Features
- Real-time accident detection using YOLO-based models
- Integration with Firebase for data storage and notifications
- Web interface for monitoring and alerts

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/max-mani/accident_detection_ai.git
   cd accident_detection_ai
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download and place required files:**
   - `ffmpeg.exe`, `ffplay.exe`, and `ffprobe.exe` (FFmpeg binaries) are required for video processing. Download them from the [official FFmpeg website](https://ffmpeg.org/download.html) and place them in the appropriate directory (e.g., `ffmpeg-7.1.1-essentials_build/bin/`).
   - `firebase-credentials.json` is required for Firebase integration. Obtain this file from your Firebase project settings and place it in the project root.

   **Note:** These files are not included in the repository due to size and security restrictions.

4. **Run the application:**
   ```bash
   python app.py
   ```

## Security Notice
- **Secrets:** The file `firebase-credentials.json` contains sensitive credentials and must be kept secure. Do not share or commit this file to any public repository.
- **Large Files:** FFmpeg binaries are not included in the repository due to GitHub's file size limitations. Download them manually as described above.

## License
This project is licensed under the MIT License. 