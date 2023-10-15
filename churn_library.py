'''

Churn_libary module
Author: Akindele Abdulrasheed
Date: Aug 6th, 2022
'''
import os


import constants as cs
from data_transform import DataTransform
from feature_train import TrainEvaluate

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if __name__ == "__main__":
    data_input = DataTransform()
    data_input.import_data(r"./data/bank_data.csv")
    data_input.perform_eda()
    data_input.encoder_helper('Churn')
    Evaluate = TrainEvaluate(data=data_input.data, label=data_input.label)
    Evaluate.perform_feature(text_size=0.3)
    Evaluate.train_models()
    Evaluate.classification_report_image()
    Evaluate.feature_importance_plot(cs.OUTPUT_PATH)
