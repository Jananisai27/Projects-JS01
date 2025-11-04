# Install dependencies before running this script:
# pip install fer opencv-python-headless mtcnn

import cv2
from fer import FER
import os

# Initialize the emotion detector
detector = FER(mtcnn=True)

# === SET YOUR VIDEO FILE PATH HERE ===
video_path = 'input_video.mp4'  # Replace with your video filename or full path

# Check if video exists
if not os.path.exists(video_path):
    print(f"Error: File '{video_path}' not found. Please check the path.")
    exit()

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create a VideoWriter object to save output
output_filename = 'output_emotion_detection.mp4'
out = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),
    10,  # Frames per second
    (frame_width, frame_height)
)

print("Processing video... Please wait.")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in each frame
    results = detector.detect_emotions(frame)
    for face in results:
        (x, y, width, height) = face["box"]
        emotions = face["emotions"]
        dominant_emotion, score = max(emotions.items(), key=lambda item: item[1])

        # Draw a rectangle and label around the detected face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f'{dominant_emotion}: {score:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

if os.path.exists(output_filename):
    print(f"\n✅ Emotion detection complete! Output saved as '{output_filename}'.")
else:
    print("❌ Error: The output file was not created.")
