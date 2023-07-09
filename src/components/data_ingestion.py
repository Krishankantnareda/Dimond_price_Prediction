import os # to create a path & save a file 
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd 
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransFormation 

## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join("artifacts","train.csv") # which data these three file will containing
    test_data_path:str=os.path.join('artifacts','test.csv')
    row_data_path:str=os.path.join('artifacts','row.csv')

# Creating a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv('notebook/Data/gemstone.csv') # Ingesting the data  os.path.join('notebooks/data','gemstone.csv'
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is comleted")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            logging.info("Till here i able to reach")

        except Exception as e:
            logging.info("Exception occurd While Ingestion the data")
            raise CustomException(e,sys)

# Run data_ingestion
if __name__=="__main__":
    obj = DataIngestion()
    train_data_path1,test_data_path2 = obj.initiate_data_ingestion()
    data_transformation = DataTransFormation()
    train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path1,test_data_path2)
    