import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
#from ydata_profiling import ProfileReport
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.entity.config_entity import ModelTrainerConfig
import tensorflow as tf

class ModelTrainer:
    def __init__(self , config: ModelTrainerConfig):
        self.config = config
    
    def log_metrics(self , y_test , y_pred , model_name:str):
        # Calculate metrics
        acc_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')  
        recall = recall_score(y_test, y_pred, average='weighted')        
        f1 = f1_score(y_test, y_pred, average='weighted') 

        metrics_file_path = self.config.results  
        with open(metrics_file_path, 'a') as file:
            file.write(f"Model Evaluation Metrics for {model_name} model:\n")
            file.write(f"Accuracy: {acc_score:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1 Score: {f1:.4f}\n")
            file.write(f"\n\n")


    def LogisticRegression(self , x_train,y_train , x_test , y_test):
        lr = LogisticRegression()
        lr.fit(x_train , y_train)

        y_pred = lr.predict(x_test)

        self.log_metrics(y_test , y_pred , "Logistic Regression")
    
    def KNN(self , x_train , y_train , x_test , y_test):
        knn = KNeighborsClassifier()
        knn.fit(x_train , y_train)

        y_pred = knn.predict(x_test)

        self.log_metrics(y_test , y_pred , "KNN")

    def RandomForest(self , x_train , y_train , x_test , y_test):
        rf = RandomForestClassifier()
        
        rf.fit(x_train , y_train)
        y_pred = rf.predict(x_test)

        model_filename = 'artifacts/model_trainer/best_model.pkl'

        # Save the trained model
        with open(model_filename, 'wb') as file:
            pickle.dump(rf, file)

        self.log_metrics(y_test , y_pred , "Random Forest Classifier")
    
    
    def neural_network(self , x_train , y_train , x_test , y_test):
        
        inputs = tf.keras.Input(shape=(x_test.shape[1],))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        batch_size = 32
        epochs = 20

        history = model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

        acc_score = model.evaluate(x_test, y_test, verbose=0)[1]
        
        metrics_file_path = self.config.results  
        with open(metrics_file_path, 'a') as file:
            file.write(f"Model Evaluation Metrics for Neural Network model:\n")
            file.write(f"Accuracy: {acc_score:.4f}\n")


    
    
    def start_model_training(self):
        x_train = np.load(self.config.preprocessed_x_train)
        x_test = np.load(self.config.preprocessed_x_test)
        y_train = np.load(self.config.preprocessed_y_train)
        y_test = np.load(self.config.preprocessed_y_test)

        self.LogisticRegression(x_train , y_train , x_test , y_test)
        self.KNN(x_train , y_train , x_test , y_test)
        self.RandomForest(x_train , y_train , x_test , y_test)
        self.neural_network(x_train , y_train , x_test , y_test)
