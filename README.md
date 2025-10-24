# Real Estate Price Prediction Analysis

A comprehensive machine learning project for predicting real estate prices in Egypt using advanced data science techniques and multiple modeling approaches.

## Project Overview

This project analyzes Egyptian real estate listings to build predictive models for property price estimation. The analysis includes data exploration, preprocessing, feature engineering, and implementation of various machine learning algorithms including traditional ML models and deep learning approaches.

## Dataset

- **Source**: [Egyptian Real Estate Listings](https://www.kaggle.com/datasets/hassankhaled21/egyptian-real-estate-listings) from Kaggle
- **Size**: ~19,924 property listings
- **Features**: Property type, location, size, bedrooms, bathrooms, payment method, and more
- **Target**: Property price prediction

## Key Features

- **Comprehensive Data Analysis**: Exploratory data analysis with interactive visualizations
- **Advanced Preprocessing**: Data cleaning, feature engineering, and transformation
- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost, Neural Networks (PyTorch)
- **Model Comparison**: Systematic evaluation using MLflow for experiment tracking
- **Production Ready**: Trained models saved for deployment and prediction

## Project Structure

```
Real-Estate-Analysis/
├── dataset/                          # Data files
│   ├── egypt_real_estate_listings.csv
│   ├── egypt_real_estate_listings_cleaned.csv
│   ├── X_train.pkl, X_test.pkl       # Training/test features
│   └── y_train.pkl, y_test.pkl       # Training/test targets
├── models/                           # Model artifacts
│   ├── best_model.pkl               # Best performing model
│   ├── scaler.pkl                   # Feature scaler
│   ├── model_columns.pkl            # Column names for model
│   ├── pca.pkl                      # PCA transformer
│   └── mlruns/                      # MLflow experiment tracking
├── notebooks/                        # Jupyter notebooks
│   ├── 00_exploration.ipynb         # Data exploration
│   ├── 01_preprocessing.ipynb       # Data cleaning & preprocessing
│   ├── 02_analytics.ipynb           # Statistical analysis
│   ├── 03_feature_engineering.ipynb # Feature creation & selection
│   ├── 04a_model_job_baseline.ipynb # Traditional ML models
│   ├── 04b_pythorch_model.ipynb     # Deep learning model
│   └── 05_prediction_notebook.ipynb # Model deployment & prediction
├── reports/                          # Generated reports
│   └── tree_map_fig.html            # Interactive visualizations
├── scripts/                          # API implementation
│   ├── price_prediction.py          # Core prediction logic
│   ├── input_schema.py              # Data validation schemas
│   └── routes/                      # API routes
├── backup/                           # Backup files
├── requirements.txt                  # Python dependencies
├── main.py                          # FastAPI application entry point
└── README.md                        # Project documentation
```

## Technologies Used

- **Python 3.12+**
- **Data Science**: pandas, numpy, matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, torch
- **Experiment Tracking**: MLflow
- **Statistical Analysis**: statsmodels
- **Web Framework**: FastAPI
- **Data Source**: Kaggle API

## Model Performance

The project implements and compares multiple machine learning approaches:

### Traditional ML Models
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble method for feature importance
- **XGBoost**: Gradient boosting for high performance
- **Decision Tree**: Interpretable model
- **K-Nearest Neighbors**: Instance-based learning

### Deep Learning
- **PyTorch Neural Network**: Multi-layer perceptron with dropout regularization

### Performance Metrics
- **R² Score**: 0.74 (best model)
- **Adjusted R²**: 0.66
- **RMSE**: 19.19%
- **Cross-validation**: 5-fold validation for robust evaluation

## Getting Started

### Prerequisites

- Python 3.12+
- Jupyter Notebook
- Git
- Kaggle API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohamedKhalifa1/Real-Estate-Price-Prediction.git
   cd Real-Estate-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API credentials**
   ```bash
   # Create kaggle.json file in ~/.kaggle/ directory
   # Download from: https://www.kaggle.com/account
   ```

4. **Download the dataset**
   - The dataset will be automatically downloaded from Kaggle when running the exploration notebook
   - Ensure you have Kaggle API credentials set up

### Usage

1. **Run the analysis pipeline**
   ```bash
   # Start Jupyter Notebook
   jupyter notebook
   ```

2. **Execute notebooks in order**:
   - `00_exploration.ipynb` - Data exploration
   - `01_preprocessing.ipynb` - Data cleaning
   - `02_analytics.ipynb` - Statistical analysis
   - `03_feature_engineering.ipynb` - Feature engineering
   - `04a_model_job_baseline.ipynb` - Traditional ML models
   - `04b_pythorch_model.ipynb` - Deep learning model
   - `05_prediction_notebook.ipynb` - Make predictions

3. **View MLflow experiments**
   ```bash
   mlflow ui --backend-store-uri models/mlruns/
   ```

4. **Run the API server**
   ```bash
   python main.py
     
   # Or using uvicorn for development
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
## API Usage

The project includes a FastAPI-based prediction service:

```python
# Example API request
import requests

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "property_type": "Apartment",
        "location": "New Cairo",
        "size": 120,
        "bedrooms": 3,
        "bathrooms": 2,
        "payment_method": "Cash"
    }
)

prediction = response.json()["prediction"]
```

## Key Insights

- **Location Impact**: Property location significantly affects pricing
- **Size Correlation**: Strong positive correlation between property size and price
- **Property Type**: Different property types show distinct pricing patterns
- **Feature Engineering**: Created meaningful features that improved model performance

## Model Deployment

The trained models are saved and ready for deployment:

```python
# Load the best model
import joblib
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Make predictions
prediction = model.predict(scaled_features)
```

## Visualizations

The project includes interactive visualizations:
- Property distribution maps
- Price distribution analysis
- Feature correlation matrices
- Model performance comparisons

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue in the repository.

## Acknowledgments

- Dataset provided by [Hassan Khaled](https://www.kaggle.com/hassankhaled21) on Kaggle
- MLflow for experiment tracking
- The open-source Python data science community

---

**Note**: This project is for educational and research purposes. Always validate model predictions with domain expertise before making real estate decisions.