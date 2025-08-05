import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import time
from collections import deque

class HandSignRecognizer:
    def __init__(self, model_path='model_isl_letters.p', labels_path='labels_alphabet.txt'):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Hands detection: ISL letters require 2 hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize model and labels
        self.model = None
        self.scaler = None
        self.labels_list = None
        self.labels_dict = None
        self.load_model(model_path, labels_path)

        # Prediction tracking
        self.prediction_history = []
        self.history_length = 5
        self.sign_recognition_counts = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_recognitions': 0,
            'accuracy_history': []
        })
        self.overall_predictions = []
        self.overall_true_labels = []
        self.prediction_confidences = []


    def load_model(self, model_path, labels_path):
        """Load the ISL letters model and its corresponding labels."""
        model_dict = pickle.load(open(model_path, 'rb'))
        self.model = model_dict['model']
        self.scaler = model_dict.get('scaler')
        try:
            with open(labels_path, 'r') as f:
                self.labels_list = [label.strip() for label in f.readlines()]
            self.labels_dict = {i: label for i, label in enumerate(self.labels_list)}
        except FileNotFoundError:
            print(f"Warning: {labels_path} not found. Using default labels.")
            self.labels_list = list('A B C D E F G H I K L M N O P Q S T U W X Y Z'.split())
            self.labels_dict = {i: label for i, label in enumerate(self.labels_list)}


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

        # Do NOT apply scaler here; apply after full feature vector is built
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
            print("No model loaded. Please check your model path.")
            return

        cap = cv2.VideoCapture(0)
        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # Draw all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Adapt to model's expected input size
                n_hands = len(results.multi_hand_landmarks)
                data_aux = []
                x_ = []
                y_ = []

                # Always build 84 features: two hands (left-to-right), or one hand + zeros
                if n_hands == 2:
                    hands_sorted = sorted(
                        results.multi_hand_landmarks,
                        key=lambda hand: np.mean([lm.x for lm in hand.landmark])
                    )
                    for hand_landmarks in hands_sorted:
                        d, xh, yh = self.process_landmarks(hand_landmarks, H, W)
                        data_aux.extend(d)
                        x_.extend(xh)
                        y_.extend(yh)
                elif n_hands == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    d, xh, yh = self.process_landmarks(hand_landmarks, H, W)
                    data_aux.extend(d)
                    x_.extend(xh)
                    y_.extend(yh)
                    data_aux.extend([0.0]*42)
                else:
                    cv2.putText(frame, "Show hand(s) for ISL letter recognition", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Hand Sign Recognition', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                # Apply scaler to the full feature vector
                n_features = getattr(self.model, 'n_features_in_', 84)
                if self.scaler is not None:
                    data_aux = self.scaler.transform([data_aux])[0]
                if n_features == 42:
                    data_aux = data_aux[:42]

                # Predict with probability
                prediction_proba = self.model.predict_proba([np.asarray(data_aux)])
                prediction = self.model.predict([np.asarray(data_aux)])
                max_proba = np.max(prediction_proba)
                smoothed_prediction = self.smooth_prediction(prediction[0])
                predicted_character = self.labels_dict.get(smoothed_prediction, 'Unknown')

                # Update prediction tracking
                self.prediction_confidences.append(max_proba)
                self.overall_predictions.append(smoothed_prediction)
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
                display_text = f"{predicted_character} ({max_proba*100:.1f}%)"
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Display quit instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow('Hand Sign Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        # Reserved for stats after usage, (still in progress)
        pass

def main():
    # change the model and label paths as needed
    recognizer = HandSignRecognizer(model_path='tuned_model.p', labels_path='labels_alphabet.txt')
    recognizer.run_recognition()

if __name__ == "__main__":
    main()
