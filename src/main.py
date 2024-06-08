import os
import torch
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from typing import List,Tuple, Sequence
from torch.utils.data import Subset
from dataHandler import ECG_DATAHANDLER
from ECG_CNN_MODEL import ECG_1D_CNN_TRAINER, EarlyStopper, MetricsHolder
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score




DATASET_PATH="./ECG_ID_dataset/physionet.org/files/ecgiddb/1.0.0/"
PYTORCH_DATASET_PATH = "./ECG_ID_dataset/pytorchdata/dataset.pt"
PYTORCH_MODEL_PATH = "./ECG_ID_dataset/pytorchdata/ECG1DCNN.pth"
TEMPLATES_DATASET_PATH = "./ECG_ID_dataset/dataset_templates.csv"
# MODEL_ACTION="TRAIN"
MODEL_ACTION="LOAD"
CREATE_TEMPLATES = False


if __name__ == "__main__":

    batch_size = 20

    my_data_handler = ECG_DATAHANDLER.gather_records_info(path=DATASET_PATH)

    #IF action==SAVE will create a new dataset and save it. Otherwise will load an existing one
    ecg_dataset, n_users=my_data_handler.transform_data_to_dataset(path=PYTORCH_DATASET_PATH, action="LOAD")
    train_set, val_set = ecg_dataset.split_dataset(train_ratio=0.8)

    #Relation between user_name and it's model label
    knowledge_dict = ecg_dataset.return_knowledge_dict()
    train_dataset = Subset(ecg_dataset, train_set)
    val_dataset = Subset(ecg_dataset, val_set)

    # Create DataLoader instances with train and validation sets
    trainingLoader = DataLoader(train_dataset, batch_size=batch_size)
    validationLoader = DataLoader(val_dataset, batch_size=batch_size,)

    print("TRAINING INSTANCES: %d. VALIDATION INSTANCES: %d"%(len(train_set),len(val_set)))

    if MODEL_ACTION == "TRAIN":
            
        model=ECG_1D_CNN_TRAINER(my_data_handler.window_size,nLabels=n_users,epochs=75,lr=0.001)

        model.trainloop(trainingLoader,validationLoader, earlyStopper=EarlyStopper(patience=5, delta=0.01, verbose=True), reg=True)
        # model.trainloop(trainingLoader,validationLoader, reg=False)


        plt.plot(model.loss_during_training,label='Training Loss')
        plt.plot(model.valid_loss_during_training,label='Validation Loss')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Loss function')
        plt.legend()
        plt.show()
        torch.save(model.state_dict(), PYTORCH_MODEL_PATH)
    else:
        model=ECG_1D_CNN_TRAINER(my_data_handler.window_size,nLabels=n_users)
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH), assign=True)

    metrics = model.eval_performance(trainingLoader)
    print("TRAINING RR", metrics.calculate_recognition_rate())
    print("TRAINING Precision",metrics.calculate_weighted_precision())
    print("TRAINING Recall",metrics.calculate_weighted_recall())
    print("TRAINING F1-Score",metrics.calculate_weighted_f1_score())

    print("")
    metrics = model.eval_performance(validationLoader)
    print("VALIDATION RR",metrics.calculate_recognition_rate())
    print("VALIDATION Precision",metrics.calculate_weighted_precision())
    print("VALIDATION Recall",metrics.calculate_weighted_recall())
    print("VALIDATION F1-Score",metrics.calculate_weighted_f1_score())

    if CREATE_TEMPLATES:
        model.create_templates(knowledge_dict, n_pass=5, savePath=TEMPLATES_DATASET_PATH, save=True, plot=True)

    ## EVALUATE THE WHOLE NET?

    # Step 1: Load the data from the CSV file without a header
    data = pd.read_csv(TEMPLATES_DATASET_PATH, header=None)

    template_ecg_dataset = my_data_handler.transform_df_to_dataset(data)
    template_loader = DataLoader(template_ecg_dataset, batch_size=5)
    metrics = model.eval_performance(template_loader)

    print("")
    print("TEMPLATES RR",metrics.calculate_recognition_rate())
    print("TEMPLATES Precision",metrics.calculate_weighted_precision())
    print("TEMPLATES Recall",metrics.calculate_weighted_recall())
    print("TEMPLATES F1-Score",metrics.calculate_weighted_f1_score())
    metrics.plot_cmc_curve()


    # EVALUATE IF ANOTHER MODEL CAN TRAIN FROM TEMPLATES:

    # Step 2: Separate features (X) and target variable (y)
    X = data.iloc[:, :-1]  # Features (all columns except the last one)
    y = data.iloc[:, -1]   # Target variable (last column)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Define the Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Define the grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [200, 300, 400],  # Number of trees in the forest
        'max_depth': [None],       # Maximum depth of the trees
        'min_samples_split': [5, 10],   # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 4, 8]      # Minimum number of samples required to be at a leaf node
    }

    # Perform cross-validation grid search TODO Stratified is giving out an error...
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=StratifiedKFold(n_splits=4, shuffle=True), scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding accuracy
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Predict on the testing set using the best model
    best_rf_classifier = grid_search.best_estimator_
    y_pred = best_rf_classifier.predict(X_test)
    scores = best_rf_classifier.predict_proba(X_test)

    # Evaluate the model
    # 85 percent ACC with 4 traning instances and 1 test 0.8536585365853658
    metrics = MetricsHolder(y_test,y_pred,scores)

    print("")
    print("RFC TEMPLATES RR",metrics.calculate_recognition_rate())
    print("RFC TEMPLATES Precision",metrics.calculate_weighted_precision())
    print("RFC TEMPLATES Recall",metrics.calculate_weighted_recall())
    print("RFC TEMPLATES F1-Score",metrics.calculate_weighted_f1_score())

    metrics.plot_cmc_curve()


