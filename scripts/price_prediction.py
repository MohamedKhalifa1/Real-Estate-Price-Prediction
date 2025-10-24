import joblib
import pandas as pd
import numpy as np

class PricePrediction:
    def __init__(self , model_path , scaler_path , model_columns_path , pca_path):
        self.cat_cols = ['type', 'available_from', 'governorate', 'city', 'district', 'compound']
        self.num_cols = ['size_sqm', 'bedrooms_num', 'bathrooms_num']
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_columns = joblib.load(model_columns_path)
        self.pca = joblib.load(pca_path)
    def preprocess(self, data: dict):
        # Convert dict to DataFrame
        df = pd.DataFrame([data])

        # Drop target column if accidentally included
        if 'price' in df.columns:
            df = df.drop(columns=['price'])

        # Scale numerical features
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=self.cat_cols)

        # Align columns with model (fill missing columns with 0)
        df = df.reindex(columns=self.model_columns, fill_value=0)
        df = self.pca.transform(df)

        return df

    def predict_price(self, data: dict):
        df = self.preprocess(data)
        prediction = self.model.predict(df)
        return np.expm1(prediction[0])
