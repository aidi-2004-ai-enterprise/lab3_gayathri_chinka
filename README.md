# Lab 3: Penguins Classification with XGBoost and FastAPI

## 🔍 Course Context

This repository was created under my **AI course (AIDI-2004)** as part of **Lab 3**. It involves building a machine learning model and serving it using a FastAPI application.

## 🧑‍💻 Project Setup & Workflow

1. I first created a GitHub repository for Lab 3.
2. Then, I cloned the repository to my local device (laptop).
3. I created and implemented the following main components:

   * `train.py` for model training and saving
   * `main.py` inside the `app/` folder for FastAPI-based prediction API
4. I followed all professor instructions and maintained all quality checks:

   * Docstrings and type hints ✅
   * Input validation using `Enum` for `sex` and `island` ✅
   * Manual one-hot encoding in both `train.py` and `main.py` ✅
   * Model and API run without error ✅
   * Matching preprocessing logic in both files ✅

## 🖥️ How I Ran the Project (Step-by-step)

```bash
# 1. Create virtual environment (only once)
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\activate  # On Windows

# 3. Install required packages (with UV or pip)
uv pip install -r requirements.txt  # or manually install fastapi, xgboost, etc.

# 4. Train the model
python train.py

# 5. Run the FastAPI server
uvicorn app.main:app --reload
```

## 🌐 FastAPI Swagger UI

Once the server is running, I tested my API at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## ✅ Test Inputs & Outputs

### 🔹 Valid Input 1

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

🔸 Output:

```json
{
  "predicted_species": "Adelie",
  "confidence_scores": {
    "Adelie": 0.9986,
    "Chinstrap": 0.0007,
    "Gentoo": 0.0006
  }
}
```

### 🔹 Valid Input 2

```json
{
  "bill_length_mm": 47.4,
  "bill_depth_mm": 14.6,
  "flipper_length_mm": 212.0,
  "body_mass_g": 4725.0,
  "year": 2009,
  "sex": "female",
  "island": "Biscoe"
}
```

🔸 Output:

```json
{
  "predicted_species": "Gentoo",
  "confidence_scores": {
    "Adelie": 0.0015,
    "Chinstrap": 0.0019,
    "Gentoo": 0.9965
  }
}
```

### 🔹 Invalid Island Input

```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "year": 2007,
  "sex": "male",
  "island": "InvalidIsland"
}
```

🔸 Output:

```json
{
  "detail": [
    {
      "loc": ["body", "island"],
      "msg": "Input should be 'Torgersen', 'Biscoe' or 'Dream'",
      "type": "enum",
      "input": "InvalidIsland"
    }
  ]
}
```

### 🔹 Invalid Sex Input

```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "year": 2007,
  "sex": "unknown",
  "island": "Torgersen"
}
```

🔸 Output:

```json
{
  "detail": [
    {
      "loc": ["body", "sex"],
      "msg": "Input should be 'male' or 'female'",
      "type": "enum",
      "input": "unknown"
    }
  ]
}
```

## 📦 Files I Created

* `train.py`: Loads Seaborn dataset, does manual one-hot encoding, trains XGBoost model, saves `.json` model
* `main.py`: Defines FastAPI endpoints with full validation and prediction logic
* `app/data/`: Stores trained model, feature order, label encoder
* `pyproject.toml`: Lists dependencies
* `README.md`: This file



## 📬 Conclusion & Contact

This project helped me understand how to integrate model training and API deployment using modern tools like XGBoost and FastAPI. Testing and debugging the validation logic helped me gain hands-on experience with enum-based constraints and real-time prediction flows.

📧 **Email:** [gayathrichinka123@gmail.com](mailto:gayathrichinka123@gmail.com)
🐙 **GitHub ID:** [gayathri18chinka](https://github.com/gayathri18chinka)
