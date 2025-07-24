# Lab 3: Penguins Classification with XGBoost and FastAPI

## Overview
This project implements a machine learning pipeline for penguin species classification using the Seaborn penguins dataset. The solution includes data preprocessing, XGBoost model training, and deployment via a FastAPI application with robust input validation and error handling.

## Features
- **Data Processing**: Manual one-hot encoding for categorical features, label encoding for target variable
- **Machine Learning**: XGBoost classifier with overfitting prevention parameters
- **API Deployment**: FastAPI application with Pydantic data validation
- **Error Handling**: Graceful error handling with clear HTTP status codes and messages
- **Logging**: Comprehensive logging for model operations and API requests
- **Input Validation**: Enum-based validation for categorical inputs (sex, island)

## Project Structure
```
├── train.py                 # Model training script
├── app/
│   ├── main.py             # FastAPI application
│   ├── data/
│   │   ├── model.json      # Trained XGBoost model
│   │   ├── label_encoder.pkl    # Label encoder for species
│   │   └── feature_columns.pkl  # Feature column order
├── pyproject.toml          # UV package management
├── app.log                 # Application logs
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- UV package manager
- Dependencies listed in `pyproject.toml`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aidi-2004-ai-enterprise/lab3_gayathri_chinka.git
cd lab3_gayathri_chinka
```

### 2. Install UV (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies
```bash
uv sync
```

## Usage

### 1. Train the Model
First, train the XGBoost model on the penguins dataset:
```bash
uv run python train.py
```

This will:
- Load and preprocess the penguins dataset
- Train an XGBoost classifier
- Evaluate model performance
- Save the model to `app/data/model.json`

### 2. Start the API Server
Launch the FastAPI application:
```bash
uv run uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

### 3. Access API Documentation
Visit the interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### POST `/predict`
Predict penguin species based on physical measurements.

**Request Body:**
```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "year": 2007,
  "sex": "male",
  "island": "Torgersen"
}
```

**Valid Values:**
- `sex`: "male" or "female"
- `island`: "Torgersen", "Biscoe", or "Dream"

**Response:**
```json
{
  "predicted_species": "Adelie",
  "confidence_scores": {
    "Adelie": 0.8234,
    "Chinstrap": 0.1234,
    "Gentoo": 0.0532
  }
}
```

### GET `/health`
Check API health status.

### GET `/`
Root endpoint with welcome message.

## Example Usage

### Successful Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "year": 2007,
  "sex": "male",
  "island": "Torgersen"
}'
```

### Error Handling Example
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "year": 2007,
  "sex": "invalid_sex",
  "island": "Torgersen"
}'
```

Returns HTTP 400 with validation error details.

## Model Performance

The XGBoost model is trained with the following parameters to prevent overfitting:
- `max_depth=3`: Limits tree depth
- `n_estimators=100`: Number of boosting rounds
- `learning_rate=0.1`: Step size shrinkage

Performance metrics are displayed during training for both training and test sets.

## Error Handling

The API provides robust error handling:
- **HTTP 400**: Input validation errors (invalid sex/island values, missing fields)
- **HTTP 422**: Pydantic validation errors (wrong data types)
- **HTTP 500**: Internal server errors

All errors include descriptive messages and valid value options.

## Logging

The application logs important events:
- Model loading and initialization
- Prediction requests and results
- Input validation errors
- Server startup and health checks

Logs are written to both console and `app.log` file.

## Demo Video

[Include your screen recording here - upload the .mp4 file to the repository]

The demo video shows:
- Successful API requests with different penguin species
- Error handling for invalid inputs
- Interactive API documentation
- Model prediction responses

## Development

### Code Structure
- `train.py`: Complete training pipeline with evaluation
- `app/main.py`: FastAPI application with validation and logging

### Dependencies Management
This project uses UV for dependency management. Key dependencies:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `xgboost`: Machine learning model
- `pandas`: Data manipulation
- `scikit-learn`: Preprocessing and metrics
- `seaborn`: Dataset loading
- `pydantic`: Data validation

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## License
This project is part of AIDI-2004 coursework.
