import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import tkinter as tk
from tkinter import ttk
import time
import pyttsx3
from collections import deque

class HandSignRecognizer:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.tts_engine = pyttsx3.init()
        self.last_spoken_time = 0
        self.high_confidence_start = None
        self.current_prediction = None
        self.prediction_queue = deque(maxlen=30)  # For 2 seconds at 15 FPS
        
        # Hands detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize model and labels as None
        self.model = None
        self.scaler = None
        self.labels_list = None
        self.labels_dict = None

        # Prediction tracking
        self.prediction_history = []
        self.history_length = 5
        
        # Reporting metrics
        self.sign_recognition_counts = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_recognitions': 0,
            'accuracy_history': []
        })
        self.overall_predictions = []
        self.overall_true_labels = []
        self.prediction_confidences = []

    def should_speak_prediction(self, prediction, confidence):
        current_time = time.time()
        
        # Add current prediction to queue
        self.prediction_queue.append((prediction, confidence))
        
        # Check if all predictions in last 2 seconds are the same and have confidence > 50%
        recent_predictions = [p for p, c in self.prediction_queue if p == prediction and c > 0.50]
        
        # If we have consistent high-confidence predictions and haven't spoken recently
        if (len(recent_predictions) >= 10 and  # Allow for some frames of error
            current_time - self.last_spoken_time > 1.0 and  # Prevent rapid repetition
            prediction != self.current_prediction):  # New prediction
            
            print(f"SPEAKING: {prediction}")  # Debug print
            self.last_spoken_time = current_time
            self.current_prediction = prediction
            return True
            
        return False

    def load_model(self, model_type):
        """Load the selected model and its corresponding labels."""
        if model_type == 1:
            model_path = 'model_alphabet.p'
            labels_path = 'labels_alphabet.txt'
        elif model_type == 2:
            model_path = 'model_words.p'
            labels_path = 'labels_words.txt'
        else:
            raise ValueError("Invalid model type")

        # Load model and scaler
        model_dict = pickle.load(open(model_path, 'rb'))
        self.model = model_dict['model']
        self.scaler = model_dict.get('scaler')

        # Load labels
        try:
            with open(labels_path, 'r') as f:
                self.labels_list = [label.strip() for label in f.readlines()]
            self.labels_dict = {i: label for i, label in enumerate(self.labels_list)}
        except FileNotFoundError:
            print(f"Warning: {labels_path} not found. Using default labels.")
            self.labels_list = list('ABCDEFGHIKLMNOPQRSTUVWXY')
            self.labels_dict = {i: label for i, label in enumerate(self.labels_list)}

    def create_start_window(self):
        """Create the initial UI window for model selection."""
        root = tk.Tk()
        root.title("Hand Sign Recognition System")
        root.geometry("600x600")
        
        style = ttk.Style()
        style.configure('TButton', padding=10, font=('Helvetica', 12))
        style.configure('TLabel', font=('Helvetica', 14))

        # Center the content
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(expand=True)

        # Welcome message
        welcome_label = ttk.Label(
            main_frame, 
            text="Welcome to Hand Sign Recognition System By Team IPR154",
            wraplength=350
        )
        welcome_label.pack(pady=20)

        # Model selection message
        model_label = ttk.Label(
            main_frame,
            text="Please select a model:",
            wraplength=350
        )
        model_label.pack(pady=10)

        def start_model(model_type):
            root.destroy()
            self.load_model(model_type)
            self.run_recognition()

        # Model selection buttons
        ttk.Button(
            main_frame,
            text="ASL Alphabets",
            command=lambda: start_model(1)
        ).pack(pady=10)

        ttk.Button(
            main_frame,
            text="ASL Words",
            command=lambda: start_model(2)
        ).pack(pady=10)

        # Quit button
        ttk.Button(
            main_frame,
            text="Quit",
            command=root.destroy
        ).pack(pady=10)

        root.mainloop()

    def process_landmarks(self, hand_landmarks, height, width):
        data_aux = []
        x_ = []
        y_ = []

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y

            x_.append(x)
            y_.append(y)

        x_min, y_min = min(x_), min(y_)
        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(hand_landmarks.landmark[i].x - x_min)
            data_aux.append(hand_landmarks.landmark[i].y - y_min)

        if self.scaler is not None:
            data_aux = self.scaler.transform([data_aux])[0]

        return data_aux, x_, y_

    def smooth_prediction(self, prediction):
        if isinstance(prediction, str):
            prediction = self.labels_list.index(prediction) if prediction in self.labels_list else -1
        
        if prediction != -1:
            self.prediction_history.append(prediction)
        
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        return max(set(self.prediction_history), key=self.prediction_history.count) if self.prediction_history else -1

    def run_recognition(self):
        if self.model is None:
            print("No model loaded. Please select a model first.")
            return

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Process landmarks
                    data_aux, x_, y_ = self.process_landmarks(hand_landmarks, H, W)

                    # Predict with probability
                    prediction_proba = self.model.predict_proba([np.asarray(data_aux)])
                    prediction = self.model.predict([np.asarray(data_aux)])
                    
                    # Get max probability and corresponding label
                    max_proba = np.max(prediction_proba)
                    
                    # Smooth prediction
                    smoothed_prediction = self.smooth_prediction(prediction[0])

                    # Get predicted character
                    predicted_character = self.labels_dict.get(smoothed_prediction, 'Unknown')

                    # Update prediction tracking
                    self.prediction_confidences.append(max_proba)
                    self.overall_predictions.append(smoothed_prediction)
                    
                    # Track sign recognition performance
                    sign_metrics = self.sign_recognition_counts[predicted_character]
                    sign_metrics['total_attempts'] += 1
                    sign_metrics['successful_recognitions'] += 1
                    sign_metrics['accuracy_history'].append(max_proba)

                    # Draw bounding box and prediction
                    x1 = max(0, int(min(x_) * W) - 10)
                    y1 = max(0, int(min(y_) * H) - 10)
                    x2 = min(W, int(max(x_) * W) + 10)
                    y2 = min(H, int(max(y_) * H) + 10)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Display prediction with accuracy (in black)
                    display_text = f"{predicted_character} ({max_proba*100:.1f}%)"
                    cv2.putText(frame, 
                              display_text, 
                              (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, 
                              (0, 0, 0), 
                              2, 
                              cv2.LINE_AA)
                    # In your run_recognition method, modify the prediction part:
                    # After your existing prediction code:
                    if max_proba > 0.50:  # 60% confidence threshold
                        try:
                            if self.should_speak_prediction(predicted_character, max_proba):
                                # print(f"Attempting to speak: {predicted_character}")  # Debug print
                                self.tts_engine.say(predicted_character)
                                self.tts_engine.runAndWait()
                                print("Speech completed")  # Debug print
                        except Exception as e:
                            print(f"Error in speech output: {e}")  # Debug print

            # Display quit instructions (in black)
            cv2.putText(frame, 
                       "Press 'q' to quit or 'r' to return to model selection", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       (0, 0, 0), 
                       2)

            # Show frame
            cv2.imshow('Hand Sign Recognition', frame)

            # Check for quit or return
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cap.release()
                cv2.destroyAllWindows()
                self.create_start_window()
                return

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        # [Previous report generation code remains the same]
        pass

def main():
    recognizer = HandSignRecognizer()
    recognizer.create_start_window()

if __name__ == "__main__":
    main()
