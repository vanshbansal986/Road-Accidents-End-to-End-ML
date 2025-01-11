import pandas as pd
import numpy as np
from src import logger
from src.config.configuration import DataCleaningConfig

class DataCleaning:
    def __init__(self , config: DataCleaningConfig):
        try:
            self.config = config
        except Exception as e:
            raise e
    
    def handle_datetime_variables(self,df: pd.DataFrame) -> pd.DataFrame:
        # handle datetime variables
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

        # extract Relevant Features
        df['Start_Year'] = df['Start_Time'].dt.year
        df['Start_Month'] = df['Start_Time'].dt.month
        df['Start_Day'] = df['Start_Time'].dt.day
        df['Start_Hour'] = df['Start_Time'].dt.hour
        df['Start_Weekday'] = df['Start_Time'].dt.weekday  # Monday=0, Sunday=6
        df['Is_Weekend'] = df['Start_Weekday'].apply(lambda x: 1 if x >= 5 else 0)

        return df
    
    def drop_columns(self,df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = ['ID' , 'Start_Time' , 'End_Time' , 'Description' , 'County', 'State' , 'Zipcode' , 'Country' , 'Timezone' , 'Airport_Code', 'End_Lat' , 'End_Lng' , 'Wind_Chill(F)' , 'Precipitation(in)', 'Street' , 'Astronomical_Twilight' , 'Sunrise_Sunset' , 'Civil_Twilight' , 'City','Nautical_Twilight', 'Weather_Timestamp']

        df.drop(columns = drop_cols , inplace = True)

        return df

    def remove_outliers(self,df: pd.DataFrame) -> pd.DataFrame:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = numerical_cols.drop('Severity')
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with the bounds
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df
    
    def start_data_cleaning(self):
        # Reading the data
        df = pd.read_csv(self.config.curr_file_path)
        
        # handle datetime variables
        df = self.handle_datetime_variables(df)

        # Removing unneccessary columns
        df = self.drop_columns(df)

        # Removing outliers
        df = self.remove_outliers(df)

        # Categorizing data into diff cols
        num_cols = df.select_dtypes(include=['number']).columns.tolist()  # Columns with numeric data types
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()  # Non-numeric columns (categorical)

        num_cols.remove('Severity')

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

        # train test split
        from sklearn.model_selection import train_test_split
        df = df.sample(400000)
        train , test = train_test_split(df , test_size=0.25 , random_state=42)

        # Saving data into train and test file path
        df.to_csv(self.config.clean_file_path)
        train.to_csv(self.config.train_file_path)
        test.to_csv(self.config.test_file_path)

        