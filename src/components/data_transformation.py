import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer # For grouping up our Categorical and numerical pipeline
from sklearn.impute import SimpleImputer # For handling missing values
from sklearn.pipeline import Pipeline # creating pipeline 
from sklearn.preprocessing import OrdinalEncoder,StandardScaler # Handling Categorical Feature category and Feature scaling

from src.exception import CustomException 
from src.logger import logging
import os
#from src.utils import save_object

@dataclass
class DataTranceformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransFormation:
    def __init__(self) -> None:
        self.Data_tranceformation_config = DataTranceformationConfig()

    def get_data_tranceformation_config(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['color', 'cut','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Ideal','Very Good','Good','Premium','Fair']
            color_categories = ['G', 'E', 'F', 'H', 'D', 'I', 'J']
            clarity_categories =['SI1','VS2','VS1','SI2','VVS2','VVS1','IF','I1']

            logging.info("Pipeline Initiated")
            # Numerical Pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputing" ,SimpleImputer(strategy="median")),
                    ("Standard",StandardScaler())
                    ]
                    )

            # Categorical Pipeline
            categorcal_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="most_frequent")),
                    ("Encoding" , OrdinalEncoder(categories=[cut_categories , color_categories,clarity_categories])),
                    ("scaler" , StandardScaler())]
                    )
            # Now i need to combine these
            preprocessor=ColumnTransformer(
            [("numerical_pipeline",numerical_pipeline,numerical_cols),
            ("categorcal_pipeline",categorcal_pipeline,categorical_cols)
            ]
            )
            return preprocessor
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_tranceformation_config()
            
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1) # X_train
            target_feature_train_df=train_df[target_column_name]                # X_test

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)     # y_train
            target_feature_test_df=test_df[target_column_name]                  #y_test
            
            ## Trnasformating using preprocessor obj
            print("_________________________________________________")
            print(input_feature_train_df['color'].value_counts())
            print("Train")
            print(input_feature_test_df['color'].value_counts())
            print("Train")
            print("_______________________________")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.Data_tranceformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.Data_tranceformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)
