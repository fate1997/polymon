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
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
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
        'random_state': 2025,
        'n_estimators': 10000,
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
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
    param['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.02, step=0.001)
    param['depth'] = trial.suggest_int('depth', 9, 15)
    param['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1.0, 5.5, step=0.5)
    param['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = 10000
    param['eval_metric'] = 'MAE'
    param['loss_function'] = 'MAE'
    param['random_state'] = 2025
    param['logging_level'] = 'Silent'

    return param


def get_tabpfn_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get TabPFN parameters for hyper-parameter tuning.
    """
    
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 8, 64),
    }

    return param