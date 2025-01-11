import pandas as pd
import numpy as np
from src import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.entity.config_entity import PreprocessingConfig

class Preprocessing:
    def __init__(self , config: PreprocessingConfig):
        try:
            self.config = config
        except Exception as e:
            raise e
    
    def transform_x(self , x_train , x_test):
        df = pd.read_csv(self.config.train_file_path)
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()  # Columns with numeric data types
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()  # Non-numeric columns (categorical)

        # List of columns to exclude from null-checks
        keep_cols = [
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Direction",
            "Wind_Speed(mph)",
            "Weather_Condition"
        ]
        
        null_int_cols = list(set(keep_cols).intersection(set(num_cols)))
        null_cat_cols = list(set(keep_cols).intersection(set(cat_cols)))


        # Define transformers
        imp_enc = ColumnTransformer(
            transformers=[
                ("num_missing", SimpleImputer(strategy="median"), null_int_cols),  # Impute missing values for numerical columns
                ("cat_imputer_ohe", Pipeline(steps=[
                    ("cat_imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values for categorical columns
                    ("ohe_trf", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))  # Apply OneHotEncoder to all categorical columns
                ]), cat_cols),  # Apply both imputation and one-hot encoding to all categorical columns
            ],
            remainder='passthrough'  # Keep other columns as they are
        )

        
        yj_trf = PowerTransformer()
        
        scaler_trf = ColumnTransformer([
            ("scaler_trf" , StandardScaler() , slice(0,40))
        ])
        
        pca = PCA(n_components=15)

        pre_pipe = Pipeline([
            ("preprocessor" , imp_enc),
            ("yj_trf" , yj_trf),
            ("scaler_trf" , scaler_trf),
            ("pca" , pca)
        ])

        x_train_trf = pre_pipe.fit_transform(x_train)
        x_test_trf = pre_pipe.transform(x_test)

        np.save(self.config.preprocessed_x_train , x_train_trf)
        np.save(self.config.preprocessed_x_test , x_test_trf)        
        joblib.dump(pre_pipe, self.config.preprocesser_obj) 


    def transform_y(self , y_train , y_test):
        
        le = LabelEncoder()
        le.fit(y_train)

        y_train = le.transform(y_train)
        y_test= le.transform(y_test)

        np.save(self.config.preprocessed_y_train , y_train)
        np.save(self.config.preprocessed_y_test , y_test)
    
    def start_preprocessing(self):
        
        train_file_path = self.config.train_file_path
        test_file_path = self.config.test_file_path

        train = pd.read_csv(train_file_path)
        test = pd.read_csv(test_file_path)

        x_train = train.drop(columns=['Severity'])
        y_train = train['Severity']
        x_test = test.drop(columns=['Severity'])
        y_test = test['Severity']

        
        self.transform_x(x_train , x_test)
        self.transform_y(y_train , y_test)

        

