import cv2
import os

# Path to the video file
video_path = "C:/company/Yolov8/input/cropped_patient26.mp4"
# Directory to save the frames
output_folder = "C:/company/Yolov8/classification/dataset/inbody/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Frame counter
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read one frame at a time
    if not ret:
        break  # Exit the loop if no more frames
    
    # Save the frame as an image file
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

# Release the video capture object
cap.release()

print(f"Extracted {frame_count} frames and saved to {output_folder}")
