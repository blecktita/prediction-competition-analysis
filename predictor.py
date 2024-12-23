"""
Amazon Product Category Prediction Service.

A Flask-based REST API service that predicts whether a product category
contains Amazon-owned products using a pre-trained XGBoost model.
"""

import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import xgboost as xgb
from flask import Flask, request, jsonify, Response
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration settings for the prediction service."""
    
    model_path: str = 'model_v1.bin'
    host: str = '0.0.0.0'
    port: int = 9696
    threshold: float = 0.72
    debug: bool = True


class ModelService:
    """Handles model loading and prediction logic."""
    
    def __init__(self, model_path: str, threshold: float):
        """
        Initialize the model service.
        
        Args:
            model_path: Path to the saved model file
            threshold: Probability threshold for positive prediction
        """
        self.threshold = threshold
        self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> None:
        """
        Load the model and vectorizer from file.
        
        Args:
            model_path: Path to the saved model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or invalid
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            with open(model_path, 'rb') as f_in:
                self.vectorizer, self.model = pickle.load(f_in)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
            
    def predict(self, category: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Make prediction for a single category.
        
        Args:
            category: Dictionary containing category features
            
        Returns:
            Tuple of (prediction probability, boolean prediction)
            
        Raises:
            ValueError: If input features are invalid
        """
        try:
            X_category = self.vectorizer.transform([category])
            X = xgb.DMatrix(
                X_category,
                feature_names=self.vectorizer.get_feature_names_out().tolist()
            )
            probability = float(self.model.predict(X)[0])
            prediction = probability > self.threshold
            
            return probability, prediction
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


class PredictionAPI:
    """Flask API for serving predictions."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the API service.
        
        Args:
            config: Configuration settings
        """
        self.config = config
        self.model_service = ModelService(
            model_path=config.model_path,
            threshold=config.threshold
        )
        self.app = Flask('amazon_check')
        self._setup_routes()
        
    def _setup_routes(self) -> None:
        """Setup API routes."""
        self.app.route('/predict', methods=['POST'])(self.predict)
        
    def predict(self) -> Response:
        """
        Prediction endpoint handler.
        
        Returns:
            JSON response with prediction results
        """
        try:
            # Validate input
            category = request.get_json()
            if not category:
                return jsonify({
                    'error': 'No input data provided'
                }), 400
                
            # Make prediction
            probability, prediction = self.model_service.predict(category)
            
            # Return results
            result = {
                'amazon_probability': probability,
                'amazon_guess': prediction
            }
            return jsonify(result)
            
        except ValueError as e:
            return jsonify({
                'error': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'error': 'Internal server error'
            }), 500

    def run(self) -> None:
        """Start the Flask application."""
        self.app.run(
            debug=self.config.debug,
            host=self.config.host,
            port=self.config.port
        )


def main():
    """Main function to run the prediction service."""
    try:
        # Initialize and run API service
        config = ModelConfig()
        service = PredictionAPI(config)
        service.run()
        
    except Exception as e:
        print(f"Failed to start service: {str(e)}")
        raise

def create_app():
    """Create and return the Flask application."""
    config = ModelConfig()
    service = PredictionAPI(config)
    return service.app

if __name__ == "__main__":
    main()