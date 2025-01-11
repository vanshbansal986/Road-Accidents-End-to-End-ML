import os
from src import logger
import pandas as pd
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self , config: DataIngestionConfig):
        try:
            self.config = config
        except Exception as e:
            raise e
    
    def store_data(self):
        # store the path of file
        curr_file_path = self.config.curr_data_path
        store_file_path = self.config.store_data_path

        # Define the size of the sample
        sample_size = 600000
        chunk_size = 10000  # Adjust based on your system's memory capacity

        # Initialize an empty list to store sampled rows
        sampled_rows = []

        # Iterate over the CSV file in chunks
        for chunk in pd.read_csv(curr_file_path, chunksize=chunk_size):
            # Randomly sample rows from the current chunk
            sampled_chunk = chunk.sample(n=min(sample_size, len(chunk)))
            sampled_rows.append(sampled_chunk)
            sample_size -= len(sampled_chunk)
            if sample_size <= 0:
                break
            
        # Concatenate all sampled chunks into a single DataFrame
        sampled_data = pd.concat(sampled_rows)

        # Save the sampled data to a new CSV file
        sampled_data.to_csv(store_file_path, index=False)