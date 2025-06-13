from flask import Flask, render_template, request, jsonify, redirect, render_template, url_for, session
from tensorflow.keras.applications import MobileNetV3Small
from extract_features import extract_features
from werkzeug.utils import secure_filename
from datetime import timedelta
from LSTMAtt import LSTMAtt
import tensorflow as tf
import tempfile
import shutil
import numpy as np
import cv2
import os
import base64


global base_model
global model

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = 'bgkhasgASHGJhs21r2@2'
app.permanent_session_lifetime = timedelta(days=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/importvideo')
def importvideo():
    return render_template('importvideo.html')


@app.route('/DetectDrowiness', methods=['POST'])
def detect_drowsiness():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'images' not in data or len(data['images']) != 30:
            return jsonify({'error': 'Invalid request: 30 images required'}), 400

        # Create temporary directory to store images
        temp_dir = tempfile.mkdtemp()
        features = []

        # Process each base64 image
        for i, img_data in enumerate(data['images']):
            # Remove base64 prefix (e.g., "data:image/jpeg;base64,")
            img_data = img_data.split(',')[1] if ',' in img_data else img_data
            # Decode base64 to image
            img_bytes = base64.b64decode(img_data)
            img_path = os.path.join(temp_dir, f'image_{i}.jpg')

            # Save image to temporary file
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            # Extract features
            feature = extract_features(img_path, base_model)
            if feature is None:
                shutil.rmtree(temp_dir)  # Clean up
                return jsonify({'error': f'Failed to extract features for image {i}'}), 500
            features.append(feature)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        # Prepare features for LSTM model (shape: 1, 30, 576)
        features = np.array(features)  # Shape: (30, 576)
        features = np.expand_dims(features, axis=0)  # Shape: (1, 30, 576)

        # Predict using LSTM model
        with tf.device('/GPU:0'):
            prediction = model.predict(features)
            result = np.argmax(prediction, axis=1)[0]  # Get class (0, 1, or 2)

        return jsonify({'result': int(result)}), 200

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/DetectDrowsinessFromVideo', methods=['POST'])
def detect_drowsiness_from_video():
    try:
        # Check if video file is in request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if not video_file or video_file.filename == '':
            return jsonify({'error': 'Invalid video file'}), 400

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, secure_filename(video_file.filename))
        video_file.save(video_path)

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Could not open video file'}), 500

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 6  # Default to 6 fps if unknown
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps  # Total duration in seconds
        segment_duration = 5  # Each segment is 5 seconds
        num_segments = int(np.ceil(duration / segment_duration))  # Number of 5-second segments

        results = []
        frame_paths = []

        for segment_idx in range(num_segments):
            # Calculate start and end times for the segment
            start_time = segment_idx * segment_duration
            end_time = min(start_time + segment_duration, duration)

            # Calculate frame indices for even sampling
            segment_frame_count = int((end_time - start_time) * fps)
            target_frames = 30  # Always sample 30 frames
            if segment_frame_count < 1:
                continue  # Skip empty segments

            # Determine frame indices to sample
            if segment_frame_count >= target_frames:
                # Evenly sample 30 frames
                step = segment_frame_count / target_frames
                frame_indices = [int(start_time * fps + i * step) for i in range(target_frames)]
            else:
                # Fewer than 30 frames, sample all and pad later
                frame_indices = [int(start_time * fps + i * (segment_frame_count / max(segment_frame_count, 1))) for i in range(segment_frame_count)]

            # Extract frames
            segment_frames = []
            for idx in frame_indices:
                if idx >= frame_count:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame to temporary file
                frame_path = os.path.join(temp_dir, f'segment_{segment_idx}_frame_{len(segment_frames)}.jpg')
                cv2.imwrite(frame_path, frame)
                segment_frames.append(frame_path)
                frame_paths.append(frame_path)

            # Pad with last frame if fewer than 30 frames
            while len(segment_frames) < target_frames:
                segment_frames.append(segment_frames[-1])

            # Extract features for segment
            features = []
            for f_path in segment_frames:
                feature = extract_features(f_path, base_model)
                if feature is None:
                    shutil.rmtree(temp_dir)
                    cap.release()
                    return jsonify({'error': f'Failed to extract features for frame {f_path}'}), 500
                features.append(feature)

            # Prepare features for LSTM model (shape: 1, 30, 576)
            features = np.array(features)  # Shape: (30, 576)
            features = np.expand_dims(features, axis=0)  # Shape: (1, 30, 576)

            # Predict using LSTM model
            with tf.device('/GPU:0'):
                prediction = model.predict(features)
                result = np.argmax(prediction, axis=1)[0]  # Get class (0, 1, or 2)
                results.append(int(result))

        # Clean up
        cap.release()
        shutil.rmtree(temp_dir)

        return jsonify({'results': results}), 200

    except Exception as e:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if 'cap' in locals():
            cap.release()
        print(f"Error processing video: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    base_model = MobileNetV3Small(weights='imagenet', include_top=False)
    input_shape = (30, 576)
    model = LSTMAtt(hidden_node=128, input_shape=input_shape, output_width=64)
    model.load("model/model_lstm_attention.keras")
    app.run(host='0.0.0.0', debug=True, port=5000)    