from datetime import datetime
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import time

# hyperparameter selection
from hyperopt import tpe, STATUS_OK, Trials, space_eval

# selected features
from utils.selected_features import Features

# model
from utils.cnn_1d import CNN1DClassifier

# hyperparameter selection
from hyperparameters.space import HpOptParametersSpace

def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict and save model outputs')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the CSV file containing the prediction result')
    return parser.parse_args()

def predict(df, features, preprocessor, used_models, random_state):
    start_time = time.time()
    
    # Set the feature name and type
    feature_name = 'gender_age_blood_karyotype_mutation'
    feature_type = features.gender_data + features.age_data + features.blood_data + \
                    features.karyotype_data + features.mutation_data

    # Get the features and rearrange the columns of the data
    X_data = df[feature_type].copy()

    # Normalize the data
    X_data[features.age_data + features.blood_data] = preprocessor.transform(X_data)

    # Convert the data type to numpy array
    X_data = X_data.values

    # Predict the probability of each model
    y_pred_probas = []
    for model_type in used_models:
        if model_type != 'cnn_1d':
            model_path = f'codes/models/{model_type}.{feature_name}.pkl'
            model = pickle.load(open(model_path, 'rb'))
            y_pred_probas.append(model.predict_proba(X_data))
        else:
            cnn1d_trials_dict = pickle.load(open(f'codes/best_trial/cnn_1d.{feature_name}.pkl', 'rb'))
            best_trial = cnn1d_trials_dict[random_state]
            best_params_dict = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in best_trial['misc']['vals'].items()}
            best_params = space_eval(getattr(HpOptParametersSpace(), f'{model_type}_cli_params'), best_params_dict)
            
            model_path = f'codes/models/{model_type}.{feature_name}.pth'
            model = CNN1DClassifier(device_name='cuda', 
                                    num_features=X_data.shape[1],
                                    num_targets=3,
                                    **best_params)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            y_pred_probas.append(model.predict_proba(X_data))

    # Load the weights of each model
    weights_path = f'codes/models/weights.{feature_name}.pkl'
    weights = pickle.load(open(weights_path, 'rb'))
    weights = np.array(weights)

    # Ensemble the probability of each model
    y_pred_proba = np.zeros_like(y_pred_probas[0])
    for i, pred in enumerate(y_pred_probas):
        y_pred_proba += weights[i] * pred
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Save the prediction result
    result_df = df.copy()
    result_df['esb_pred'] = y_pred

    # Integrate the Ensemble (ML) and the ELN 2022 (clinical treatment)
    result_df.loc[(result_df['ELN_2022'] == 0) & (result_df['esb_pred'] == 0), 'esb_eln_pred'] = 0
    result_df.loc[(result_df['ELN_2022'] == 0) & (result_df['esb_pred'] == 1), 'esb_eln_pred'] = 1
    result_df.loc[(result_df['ELN_2022'] == 0) & (result_df['esb_pred'] == 2), 'esb_eln_pred'] = 1
    result_df.loc[(result_df['ELN_2022'] == 1) & (result_df['esb_pred'] == 0), 'esb_eln_pred'] = 0
    result_df.loc[(result_df['ELN_2022'] == 1) & (result_df['esb_pred'] == 1), 'esb_eln_pred'] = 1
    result_df.loc[(result_df['ELN_2022'] == 1) & (result_df['esb_pred'] == 2), 'esb_eln_pred'] = 2
    result_df.loc[(result_df['ELN_2022'] == 2) & (result_df['esb_pred'] == 0), 'esb_eln_pred'] = 1
    result_df.loc[(result_df['ELN_2022'] == 2) & (result_df['esb_pred'] == 1), 'esb_eln_pred'] = 1
    result_df.loc[(result_df['ELN_2022'] == 2) & (result_df['esb_pred'] == 2), 'esb_eln_pred'] = 2
    
    result_df['esb_eln_pred'] = result_df['esb_eln_pred'].astype(int)
    
    duration = time.time() - start_time
    print(f'Prediction is done. Duration: {duration:.2f} seconds.')
    return result_df


if __name__ == '__main__':
    total_start_time = time.time()

    # Parse the arguments
    args = parse_arguments()

    # Load the data
    df = pd.read_csv(args.data_path, sep=',', encoding='utf-8')

    # Load the features
    features = Features()

    # Load the preprocessor
    preprocessor = pickle.load(open('codes/models/preprocessor.pkl', 'rb'))

    # Load the used models
    used_models = {
        'lr'    : 'lr_best_trials_dict',
        'knn'   : 'knn_best_trials_dict',
        'svc'   : 'svc_best_trials_dict',
        'rf'    : 'rf_best_trials_dict',
        'xgb'   : 'xgb_best_trials_dict',
        'lgb'   : 'lgb_best_trials_dict',
        'cnn_1d': 'cnn_1d_best_trials_dict'
    }

    # Select model by the validation performance (c-index = 0.7)
    random_state = 22

    # Predict
    result_df = predict(df, features, preprocessor, used_models, random_state)

    # # Named the prediction result by datetime
    # path = f'./results/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}.csv'
    # result_df.to_csv(path, sep=',', index=False)

    # Save the prediction result
    result_df.to_csv(args.output_path, sep=',', index=False)

    total_duration = time.time() - total_start_time
    print(f'Total duration: {total_duration:.2f} seconds.')