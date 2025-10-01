import numpy as np
import pickle
import tensorflow as tf
from autils import load_data

class HandwrittenDigitClassifier:
    """
    A simple classification system for handwritten digit recognition.
    This system loads a pre-trained model and provides prediction functionality.
    """
    
    def __init__(self, model_path='trained_model.pkl'):
        """
        Initialize the classifier by loading the trained model.
        
        Args:
            model_path (str): Path to the pickle file containing the trained model
        """
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model from pickle file."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found.")
            print("Please make sure the model has been trained and saved first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, image_data):
        """
        Predict the class of a handwritten digit image.
        
        Args:
            image_data (numpy.ndarray): Image data as a flattened array of shape (400,)
                                      or reshaped array of shape (1, 400)
        
        Returns:
            tuple: (prediction_probability, predicted_class)
                   - prediction_probability: Raw probability from model (0-1)
                   - predicted_class: Binary classification (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize the classifier first.")
        
        # Ensure input is in the correct format
        if image_data.shape == (400,):
            image_data = image_data.reshape(1, 400)
        elif image_data.shape != (1, 400):
            raise ValueError(f"Input shape {image_data.shape} is invalid. Expected (400,) or (1, 400)")
        
        # Make prediction
        prediction_prob = self.model.predict(image_data, verbose=0)[0][0]
        
        # Apply threshold to get binary classification
        predicted_class = 1 if prediction_prob >= 0.5 else 0
        
        return prediction_prob, predicted_class
    
    def predict_batch(self, image_batch):
        """
        Predict classes for a batch of images.
        
        Args:
            image_batch (numpy.ndarray): Batch of image data of shape (n_samples, 400)
        
        Returns:
            tuple: (prediction_probabilities, predicted_classes)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize the classifier first.")
        
        if len(image_batch.shape) != 2 or image_batch.shape[1] != 400:
            raise ValueError(f"Input shape {image_batch.shape} is invalid. Expected (n_samples, 400)")
        
        # Make predictions
        prediction_probs = self.model.predict(image_batch, verbose=0).flatten()
        
        # Apply threshold to get binary classifications
        predicted_classes = (prediction_probs >= 0.5).astype(int)
        
        return prediction_probs, predicted_classes
    
    def evaluate_on_test_data(self, X_test=None, y_test=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray, optional): Test features. If None, loads from data files.
            y_test (numpy.ndarray, optional): Test labels. If None, loads from data files.
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, etc.
        """
        if X_test is None or y_test is None:
            print("Loading test data...")
            X_test, y_test = load_data()
        
        # Make predictions
        prediction_probs, predicted_classes = self.predict_batch(X_test)
        
        # Flatten y_test for comparison
        y_test_flat = y_test.flatten()
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == y_test_flat)
        
        # Calculate precision and recall
        true_positives = np.sum((predicted_classes == 1) & (y_test_flat == 1))
        false_positives = np.sum((predicted_classes == 1) & (y_test_flat == 0))
        false_negatives = np.sum((predicted_classes == 0) & (y_test_flat == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_samples': len(y_test_flat),
            'correct_predictions': np.sum(predicted_classes == y_test_flat)
        }
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return "Model not loaded"
        
        return {
            'model_type': type(self.model).__name__,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'layers': len(self.model.layers)
        }


def main():
    """
    Example usage of the HandwrittenDigitClassifier.
    """
    print("Handwritten Digit Classification System")
    print("=" * 40)
    
    try:
        # Initialize the classifier
        classifier = HandwrittenDigitClassifier()
        
        # Display model information
        print("\nModel Information:")
        model_info = classifier.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Load test data
        print("\nLoading test data...")
        X, y = load_data()
        
        # Evaluate the model
        print("\nEvaluating model on test data...")
        metrics = classifier.evaluate_on_test_data(X, y)
        
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Example prediction on a single sample
        print("\nExample Prediction:")
        sample_image = X[0]  # First sample
        prob, pred_class = classifier.predict(sample_image)
        actual_class = y[0][0]
        
        print(f"  Sample prediction probability: {prob:.4f}")
        print(f"  Predicted class: {pred_class}")
        print(f"  Actual class: {actual_class}")
        print(f"  Prediction correct: {pred_class == actual_class}")
        
        # Example batch prediction
        print("\nBatch Prediction Example (first 5 samples):")
        batch_probs, batch_preds = classifier.predict_batch(X[:5])
        batch_actuals = y[:5].flatten()
        
        for i in range(5):
            print(f"  Sample {i+1}: Prob={batch_probs[i]:.4f}, Pred={batch_preds[i]}, Actual={batch_actuals[i]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the model file 'trained_model.pkl' exists and is valid.")


if __name__ == "__main__":
    main()
