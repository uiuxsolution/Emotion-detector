# Real-time Facial Emotion Recognition

This project implements a real-time facial emotion recognition system using OpenCV and TensorFlow. The system can detect faces in video streams and classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Real-time face detection using OpenCV's Haar Cascade Classifier
- Emotion classification using a Convolutional Neural Network (CNN)
- Pre-trained model for immediate use
- Live video feed processing
- Visual feedback with bounding boxes and emotion labels
- Probability scores for detected emotions

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- NumPy

## Installation

1. Clone this repository or download the source code

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment if you haven't already:
   ```bash
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Run the emotion recognition program:
   ```bash
   python emotion_recognition.py
   ```

3. The program will open your webcam and start detecting emotions in real-time

4. Press 'q' to quit the program

## How It Works

The system works in several steps:

1. Face Detection: Uses OpenCV's Haar Cascade Classifier to detect faces in each frame
2. Preprocessing: Converts detected face regions to grayscale and resizes them to 48x48 pixels
3. Emotion Recognition: Feeds the processed face image through a CNN model to predict emotions
4. Visualization: Draws bounding boxes around detected faces and displays the predicted emotion with its probability

## Model Architecture

The emotion recognition model uses a CNN architecture with the following layers:
- Multiple Convolutional layers with ReLU activation
- MaxPooling layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for final classification

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Pre-trained weights are sourced from the face_classification project
- Uses the FER2013 dataset architecture for emotion recognition