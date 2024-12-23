# Amazon Competition Predictor

A machine learning service that predicts Amazon's presence in product categories based on various market metrics. The service includes a Flask-based REST API for serving predictions and a Streamlit web interface for easy interaction.

## Overview

This project helps sellers analyze product categories on e-commerce platforms by predicting whether Amazon is likely to be present as a competitor. It uses an XGBoost model trained on historical market data to make these predictions.

### Features

- **Machine Learning Model**: Pre-trained XGBoost model for predicting Amazon's presence
- **REST API**: Flask-based API for serving predictions
- **Web Interface**: Streamlit application for easy interaction with the prediction service
- **Docker Support**: Containerized deployment for consistent runtime environment
- **Test Scenarios**: Pre-defined test cases covering various market situations

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Docker
- Poetry (for dependency management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/blecktita/prediction-competition-analysis.git
   cd prediction-competition-analysis
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

### Running the Service

#### Using Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t amazon-competition-predictor .
   ```

2. Run the container:
   ```bash
   docker run -d -p 9696:9696 amazon-competition-predictor
   ```

#### Using Local Python Environment

1. Start the prediction service:
   ```bash
   poetry run python predictor.py
   ```

2. Start the Streamlit interface:
   ```bash
   poetry run streamlit run app.py
   ```

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8501`
2. Configure the Docker image in the sidebar if needed
3. Choose between:
   - Custom Input: Enter your own category metrics
   - Sample Categories: Use pre-defined test scenarios

### API Usage

The service exposes a REST API endpoint at `/predict`. Here's an example request:

```python
import requests

data = {
    "item_count": 3000,
    "vol_purchase_total": 1700000.0,
    "high_rating_perc": 0.75,
    "rating_mean": 3.2,
    "rating_std": 0.8,
    "best_seller_count": 112
}

response = requests.post('http://localhost:9696/predict', json=data)
prediction = response.json()
```

## Input Features

- `item_count`: Number of items in the category
- `vol_purchase_total`: Total purchase volume in dollars
- `high_rating_perc`: Percentage of items with high ratings (0.0-1.0)
- `rating_mean`: Average product rating (1.0-5.0)
- `rating_std`: Standard deviation of ratings
- `best_seller_count`: Number of best-selling items

## Model Details

The service uses an XGBoost model trained on historical market data. The model:
- Uses a probability threshold of 0.72 for positive predictions
- Returns both probability and binary prediction
- Considers multiple market metrics for predictions

## Testing

The project includes pre-defined test scenarios covering various market situations:
- Large Electronics Category
- Niche Craft Category
- Emerging Tech Category
- Luxury Category
- And more...

Run the test suite:
```bash
poetry run python test_prediction_service.py
```

## Project Structure

```
├── app.py                      # Streamlit app
├── predictor.py                # Flask API service
├── test_prediction_service.py  # Test scenarios
├── model_v1.bin               # Pre-trained model
├── Dockerfile                 
├── pyproject.toml            
└── README.md                 
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request