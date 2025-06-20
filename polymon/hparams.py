from typing import Any, Dict, Literal

import optuna


def get_hparams(
    trial: optuna.Trial, 
    model: Literal['xgb', 'rf', 'lgbm', 'catboost', 'tabpfn']
) -> Dict[str, float]:
    """Get hyper-parameters for a model.
    """
    
    if model == 'xgb':
        return get_xgb_hparams(trial)
    elif model == 'rf':
        return get_rf_hparams(trial)
    elif model == 'lgbm':
        return get_lgbm_hparams(trial)
    elif model == 'catboost':
        return get_catboost_hparams(trial)
    elif model == 'tabpfn':
        return get_tabpfn_hparams(trial)
    else:
        raise ValueError(f'Invalid model: {model}')


def get_xgb_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get XGBoost parameters for hyper-parameter tuning.
    https://www.kaggle.com/code/alisultanov/regression-xgboost-optuna
    """

    param = {
        'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        'sampling_method': 'gradient_based',
        'lambda': trial.suggest_loguniform('lambda', 7.0, 17.0),
        'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0),
        'eta': trial.suggest_categorical('eta', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'gamma': trial.suggest_categorical('gamma', [18, 19, 20, 21, 22, 23, 24, 25]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 8, 600),  
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),  
        'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
        'random_state': 42
    }
    
    return param


def get_rf_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get Random Forest parameters for hyper-parameter tuning.
    https://www.kaggle.com/code/mustafagerme/optimization-of-random-forest-model-using-optuna
    """
    
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "n_jobs": -1,
    }
    return param


def get_lgbm_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get LightGBM parameters for hyper-parameter tuning.
    https://www.kaggle.com/code/hamzaghanmi/lgbm-hyperparameter-tuning-using-optuna
    """
    
    param = {
        'metric': 'mae', 
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }

    return param


def get_catboost_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get CatBoost parameters for hyper-parameter tuning.
    https://www.kaggle.com/code/tomokikmogura/catboost-hyperparameters-tuning-with-optuna
    """
    
    param = {}
    param['learning_rate'] = trial.suggest_discrete_uniform("learning_rate", 0.001, 0.02, 0.001)
    param['depth'] = trial.suggest_int('depth', 9, 15)
    param['l2_leaf_reg'] = trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5)
    param['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = 10000
    param['use_best_model'] = True
    param['eval_metric'] = 'MAE'
    param['od_type'] = 'iter'
    param['od_wait'] = 20
    param['random_state'] = 42
    param['logging_level'] = 'Silent'

    return param


def get_tabpfn_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get TabPFN parameters for hyper-parameter tuning.
    """
    
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 8, 32),
        "softmax_temperature": trial.suggest_float("softmax_temperature", 0.1, 1.0),
        "average_before_softmax": trial.suggest_categorical("average_before_softmax", [True, False]),
    }

    return param