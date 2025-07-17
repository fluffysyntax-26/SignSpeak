import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm  

def process_hand_landmarks(data_dir, output_file='data.pickle', min_confidence=0.5, max_hands=2): # Adjusted default min_confidence
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=min_confidence, max_num_hands=max_hands)

    data = []
    labels = []
    processing_errors = []

    # Get all the image paths for the progress bar
    image_data = []
    for label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, label)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue

        # Collect image paths for the progress bar
        for img_filename in os.listdir(class_path):
            img_path = os.path.join(class_path, img_filename)
            image_data.append((img_path, label))

    # Process images with a progress bar
    for img_path, label in tqdm(image_data, desc="Processing images", unit="image"):
        try:
            # Read and convert image
            img = cv2.imread(img_path)
            if img is None:
                processing_errors.append(f"Could not read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe
            results = hands.process(img_rgb)

            # Check if hand landmarks were detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract normalized coordinates
                    x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]

                    # Normalize coordinates relative to the bounding box
                    data_aux = []
                    x_min, y_min = min(x_coordinates), min(y_coordinates)
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - x_min)
                        data_aux.append(hand_landmarks.landmark[i].y - y_min)

                    # Ensure consistent data length (in case of partial detection)
                    if len(data_aux) > 0:
                        data.append(data_aux)
                        labels.append(label)
            else:
                # Consider logging only a sample of these to avoid log file bloat.
                if len(processing_errors) < 100:  # Example: Log only the first 100
                    processing_errors.append(f"No hand landmarks detected in: {img_path}")
                processing_errors.append(f"No hand landmarks detected in: {img_path}")

        except Exception as e:
            processing_errors.append(f"Error processing {img_path}: {str(e)}")

    # Save processed data
    with open(output_file, 'wb') as f:
        pickle.dump({
            'data': data, 
            'labels': labels
        }, f)

    # Provide processing summary
    print(f"Processed {len(data)} images")
    print(f"Number of processing errors: {len(processing_errors)}")
    
    # Log detailed errors if any
    if processing_errors:
        with open('processing_errors.log', 'w') as error_log:
            error_log.write('\n'.join(processing_errors))

    return data, labels

# Run the processing
if __name__ == "__main__":
    DATA_DIR = './isl_data' # Corrected to match collection script
    process_hand_landmarks(DATA_DIR)
