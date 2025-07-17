import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
)
from sklearn.preprocessing import StandardScaler

class TqdmJoblibCallback:
    """Custom callback to show progress in RandomForest training with tqdm."""
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training Progress", unit="tree")

    def __call__(self, _):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()

def train_hand_sign_classifier(data_path='data.pickle', save_path='model.p'):
    """
    Trains a RandomForestClassifier on hand sign data and saves the trained model.
    """
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # Preprocessing
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        data_scaled, labels, 
        test_size=0.2, 
        shuffle=True, 
        stratify=labels, 
        random_state=42
    )

    # Model Training with Progress Bar
    n_estimators = 100  # Number of trees
    tqdm_callback = TqdmJoblibCallback(total=n_estimators)

    model = RandomForestClassifier(
        n_estimators=n_estimators,  # Number of trees
        max_depth=None,            # Full depth
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  # Utilize all CPU cores
    )

    with tqdm_callback.pbar as pbar:
        # Parallelize tree building
        model.fit(x_train, y_train)
    tqdm_callback.close()

    # Predictions
    y_predict = model.predict(x_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_predict)  # Corrected typo: accuarcy -> accuracy
    print(f'Accuracy: {accuracy * 100:.2f}% of samples were classified correctly!')

    # Detailed Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_predict))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(model, data_scaled, labels, cv=5, n_jobs=-1)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    # Feature Importance
    feature_importance = model.feature_importances_
    plt.figure(figsize=(10,6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Save the model
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler
        }, f)

    return model, accuracy

if __name__ == "__main__":
    train_hand_sign_classifier()
