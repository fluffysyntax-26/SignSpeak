import os
import cv2
import time
import random  # Import the random module
# "pip install python-opencv to import cv2"

ISL_SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
    'V', 'W', 'X', 'Y'
]

DATA_DIR = './isl_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DATASET_SIZE = 1000  

def collect_sign_images():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    for sign in ISL_SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

    for sign in ISL_SIGNS:
        print(f'Preparing to collect data for sign: {sign}')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Add a bounding box visualization
            cv2.rectangle(frame, (100, 100), (540, 480), (255, 0, 0), 2)  # Draw a blue rectangle

            cv2.putText(frame, f'Prepare to show sign: {sign}', (50, 80),

                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Press "Q" when ready', (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('ISL Sign Collection', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # Add a short countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.rectangle(frame, (100, 100), (540, 480), (255, 0, 0), 2)
            cv2.putText(frame, f'Capturing in {i}...', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('ISL Sign Collection', frame)
            cv2.waitKey(1000)

            if not ret:
                print("Failed to grab frame")
                break

            current_time = time.time()
            if current_time - start_time >= 0.1:
                filename = os.path.join(DATA_DIR, sign, f'{counter}.jpg')
                cv2.imwrite(filename, frame)
                counter += 1
                start_time = current_time

        counter = 0
        start_time = time.time()
        while counter < DATASET_SIZE:
            ret, frame = cap.read()

            cv2.rectangle(frame, (100, 100), (540, 480), (255, 0, 0), 2)
            if not ret:
                print("Failed to grab frame")
                break

            current_time = time.time()
            if current_time - start_time >= 0.1:
                filename = os.path.join(DATA_DIR, sign, f'{counter}.jpg')
                cv2.imwrite(filename, frame)
                counter += 1
                start_time = current_time

            if counter == 500:
                print(f"Halfway there! Please switch hands for sign: {sign}.")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break

                    cv2.rectangle(frame, (100, 100), (540, 480), (0, 255, 0), 2)  # Change to green for hand switch

                    cv2.putText(frame, f"Switch hands for sign: {sign}", (50, 80),



                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, "Press 'Q' to continue", (100, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('ISL Sign Collection', frame)
                    if cv2.waitKey(25) == ord('q'):
                        break 

            # Conditionally display the capturing message to reduce flicker
            if random.randint(0, 5) == 0:  # Only update the text every 5 frames on average
                display_text = f'Capturing {sign}: {counter}/{DATASET_SIZE}'
            cv2.putText(frame, display_text, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow('ISL Sign Collection', frame)

            
            if cv2.waitKey(25) == ord('q'):
                break 
            if cv2.waitKey(25) == ord('e'): 
                cap.release()
                cv2.destroyAllWindows()
                return 

    print("Collection complete!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_sign_images()
