import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm  

def process_hand_landmarks(data_dir, output_file='data.pickle', min_confidence=0.3, max_hands=2): #You can change the max hands accordingly
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=min_confidence, max_num_hands=max_hands)

    data = []
    labels = []
    processing_errors = []

    # Get all the image paths for the progress bar
    image_paths = []
    for label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, label)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue

        # Collect image paths for the progress bar
        for img_filename in os.listdir(class_path):
            img_path = os.path.join(class_path, img_filename)
            image_paths.append((img_path, label))

    # Process images with a progress bar
    for img_path, label in tqdm(image_paths, desc="Processing images", unit="image"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                processing_errors.append(f"Could not read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Support both single and double hand gestures
            if results.multi_hand_landmarks:
                n_hands = len(results.multi_hand_landmarks)
                data_aux = []
                if n_hands == 2:
                    # Sort hands left-to-right by mean x
                    hands_sorted = sorted(
                        results.multi_hand_landmarks,
                        key=lambda hand: np.mean([lm.x for lm in hand.landmark])
                    )
                    for hand_landmarks in hands_sorted:
                        x_ = [landmark.x for landmark in hand_landmarks.landmark]
                        y_ = [landmark.y for landmark in hand_landmarks.landmark]
                        x_min, y_min = min(x_), min(y_)
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(hand_landmarks.landmark[i].x - x_min)
                            data_aux.append(hand_landmarks.landmark[i].y - y_min)
                elif n_hands == 1:
                    # One hand: fill the other half with zeros
                    hand_landmarks = results.multi_hand_landmarks[0]
                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]
                    x_min, y_min = min(x_), min(y_)
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - x_min)
                        data_aux.append(hand_landmarks.landmark[i].y - y_min)
                    # Pad with 42 zeros for the missing hand
                    data_aux.extend([0.0]*42)
                else:
                    processing_errors.append(f"No hand landmarks detected in: {img_path}")
                    continue
                # Should always be 84 features
                if len(data_aux) == 84:
                    data.append(data_aux)
                    labels.append(label)
                else:
                    processing_errors.append(f"Incorrect landmark count in: {img_path}")
            else:
                processing_errors.append(f"No hand landmarks detected in: {img_path}")

        except Exception as e:
            processing_errors.append(f"Error processing {img_path}: {str(e)}")

    # Save processed data
    with open(output_file, 'wb') as f:
        pickle.dump({
            'data': data, 
            'labels': labels
        }, f)

    # Print processing summary
    print(f"Processed {len(data)} images")
    print(f"Number of processing errors: {len(processing_errors)}")
    
    # Optionally, log detailed errors
    if processing_errors:
        with open('processing_errors.log', 'w') as error_log:
            error_log.write('\n'.join(processing_errors))

    return data, labels

# Run the processing
if __name__ == "__main__":
    DATA_DIR = './data'
    process_hand_landmarks(DATA_DIR)
