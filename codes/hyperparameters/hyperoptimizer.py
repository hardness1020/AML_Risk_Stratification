import os
import pickle
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from utils.cnn_1d import CNN1DClassifier
from hyperopt import fmin, STATUS_OK


class HpOpt(object):
    def __init__(self, X_train, X_test, y_train, y_test, loss_func, random_state, max_evals, feature_name, best_trial_dir_path, model_dir_path):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.loss_func = loss_func
        self.random_state = random_state
        self.max_evals = max_evals
        self.feature_name = feature_name

        self.best_trial_dir_path = best_trial_dir_path
        self.model_dir_path = model_dir_path

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            best = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            return best, trials
        except Exception as e:
            print(e)
    
    def store_best_trial(self, best_trail, file_name, feature_name, compare=True):
        best_trail = {self.random_state: best_trail}

        if not os.path.exists(f'{self.best_trial_dir_path}'):
            os.makedirs(f'{self.best_trial_dir_path}')
        
        best_trail_path = f'{self.best_trial_dir_path}/{file_name}.{feature_name}.pkl'
        if compare:
            # compare with previous best trial
            if os.path.exists(best_trail_path):
                pre_best_trial = pickle.load(open(best_trail_path, 'rb'))
                if self.random_state in pre_best_trial.keys():
                    if best_trail[self.random_state]['result']['loss'] < pre_best_trial[self.random_state]['result']['loss']:
                        print('Update best trial')
                        pre_best_trial.update(best_trail)
                        pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
                        return
                    else:
                        print('No update best trial')
                        pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
                else:
                    print('Add best trial')
                    pre_best_trial.update(best_trail)
                    pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
            else:
                print('Initialize best trial')
                pickle.dump(best_trail, open(best_trail_path, 'wb'))

        else:
            if os.path.exists(best_trail_path):
                print('Update best trial')
                pre_best_trial = pickle.load(open(best_trail_path, 'rb'))
                pre_best_trial.update(best_trail)
                pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
            else:
                print('Create best trial')
                pickle.dump(best_trail, open(best_trail_path, 'wb'))

    def store_best_model(self, best_trail, file_name, feature_name):
        tid = best_trail['tid']
        
        # remove old model
        if os.path.exists(f'{self.model_dir_path}/{file_name}.{feature_name}.pth'):
            os.remove(f'{self.model_dir_path}/{file_name}.{feature_name}.pth')

        # rename best model and delete the others
        if os.path.exists(f'{self.model_dir_path}/{file_name}.{feature_name}.{tid}.pth'):
            os.rename(f'{self.model_dir_path}/{file_name}.{feature_name}.{tid}.pth', 
                      f'{self.model_dir_path}/{file_name}.{feature_name}.pth')
            
            # remove the other models
            for file in os.listdir(f'{self.model_dir_path}'):
                if file != f'{file_name}.{feature_name}.pth':
                    os.remove(f'{self.model_dir_path}/{file}')
        else:
            raise Exception('No such model file')

    def lr_cli(self, params):
        # LogisticRegression
        cli = LogisticRegression(**params)
        fit_params = {}
        return self.train_cli(cli, fit_params)
    
    def knn_cli(self, params):
        # KNeighborsClassifier
        cli = KNeighborsClassifier(**params)
        fit_params = {}
        return self.train_cli(cli, fit_params)

    def svc_cli(self, params):
        # SVC
        cli = SVC(**params)
        fit_params = {}
        return self.train_cli(cli, fit_params)

    def rf_cli(self, params):
        # RandomForestClassifier
        cli = RandomForestClassifier(**params)
        fit_params = {}
        return self.train_cli(cli, fit_params)

    def xgb_cli(self, params):
        # XGBClassifier gpu version
        cli = XGBClassifier(gpu_id=0, tree_method='gpu_hist', eval_metric='mlogloss',
                            early_stopping_rounds=5, **params) 
        fit_params = {'eval_set': [(self.X_test, self.y_test)], 'verbose': False}
        return self.train_cli(cli, fit_params)
    
    def lgb_cli(self, params):
        # lightGBM gpu version
        # cli = lgb.LGBMClassifier(device='gpu', **params)
        cli = lgb.LGBMClassifier(**params)
        fit_params = {'eval_set': [(self.X_test, self.y_test)], 'verbose': False}
        return self.train_cli(cli, fit_params)
    
    def cnn_1d_cli(self, params):
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CNN1DClassifier(device_name=device_name, 
                                num_features=self.X_train.shape[1], 
                                num_targets=3, # set manually
                                **params)
        fit_params = {'eval_set': [(self.X_test, self.y_test)]}
        loss_and_status = self.train_cli(model, fit_params)
        model.save_model(path=f'{self.model_dir_path}',
                         feature_name=self.feature_name,
                         max_evals=self.max_evals)
        return loss_and_status
    
    def train_cli(self, cli, fit_params):
        cli.fit(self.X_train, self.y_train, **fit_params)
        pred = cli.predict_proba(self.X_test)
        loss = self.loss_func(self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}
    