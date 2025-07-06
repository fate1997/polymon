from typing import Any, Dict, Literal

import optuna


HPARAMS_REGISTRY = {}

def register_hparams(model: str):
    def decorator(func):
        HPARAMS_REGISTRY[model] = func
        return func
    return decorator

def get_hparams(
    trial: optuna.Trial, 
    model: str,
) -> Dict[str, float]:
    """Get hyper-parameters for a model.
    """
    if model not in HPARAMS_REGISTRY:
        raise ValueError(f'Invalid model: {model}')
    return HPARAMS_REGISTRY[model](trial)


@register_hparams('xgb')
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


@register_hparams('rf')
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


@register_hparams('lgbm')
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


@register_hparams('catboost')
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


@register_hparams('tabpfn')
def get_tabpfn_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get TabPFN parameters for hyper-parameter tuning.
    """
    
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 8, 64),
        "softmax_temperature": trial.suggest_categorical('softmax_temperature', [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
        "average_before_softmax": trial.suggest_categorical('average_before_softmax', [True, False]),
        "ignore_pretraining_limits": True,
    }
    return param


@register_hparams('gatv2')
def get_gatv2_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get GATv2 parameters for hyper-parameter tuning.
    """
    
    param = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 64, step=16),
        "num_layers": trial.suggest_int("num_layers", 2, 5, step=1),
        "num_heads": trial.suggest_int("num_heads", 2, 8, step=2),
        "pred_hidden_dim": trial.suggest_int("pred_hidden_dim", 16, 256, step=16),
        "pred_dropout": trial.suggest_float("pred_dropout", 0.0, 0.5),
        "pred_layers": trial.suggest_int("pred_layers", 1, 3, step=1),
    }
    return param


@register_hparams('attentivefp')
def get_attentivefp_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get AttentiveFP parameters for hyper-parameter tuning.
    """
    
    param = {
        "in_channels": trial.suggest_int("in_channels", 16, 256, step=16),
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, step=16),
        "num_layers": trial.suggest_int("num_layers", 2, 4, step=1),
        "num_timesteps": trial.suggest_int("num_timesteps", 2, 4, step=1),
    }
    return param


@register_hparams('dimenetpp')
def get_dimenetpp_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get DimeNet++ parameters for hyper-parameter tuning.
    """
    
    param = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, step=16),
        "num_layers": trial.suggest_int("num_layers", 2, 4, step=1),
        "int_emb_size": trial.suggest_int("int_emb_size", 8, 64, step=8),
        "basis_emb_size": trial.suggest_int("basis_emb_size", 4, 16, step=4),
        "out_emb_channels": trial.suggest_int("out_emb_channels", 16, 256, step=16),
        "num_spherical": trial.suggest_int("num_spherical", 5, 10, step=1),
        "num_radial": trial.suggest_int("num_radial", 4, 12, step=2),
        "cutoff": trial.suggest_float("cutoff", 2.0, 5.0, step=0.5),
        "max_num_neighbors": trial.suggest_int("max_num_neighbors", 16, 64, step=1),
    }
    return param


@register_hparams('gin')
def get_gin_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get GIN parameters for hyper-parameter tuning.
    """
    
    param = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, step=16),
        "num_layers": trial.suggest_int("num_layers", 2, 4, step=1),
        "pred_hidden_dim": trial.suggest_int("pred_hidden_dim", 16, 256, step=16),
        "pred_dropout": trial.suggest_float("pred_dropout", 0.0, 0.5),
        "pred_layers": trial.suggest_int("pred_layers", 1, 3, step=1),
        "n_mlp_layers": trial.suggest_int("n_mlp_layers", 1, 3, step=1),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }
    return param


@register_hparams('pna')
def get_pna_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get PNA parameters for hyper-parameter tuning.
    """
    
    param = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, step=16),
        "num_layers": trial.suggest_int("num_layers", 2, 4, step=1),
        "towers": trial.suggest_categorical("towers", [1, 2, 4, 8]),
        "pred_hidden_dim": trial.suggest_int("pred_hidden_dim", 16, 256, step=16),
        "pred_dropout": trial.suggest_float("pred_dropout", 0.0, 0.5),
        "pred_layers": trial.suggest_int("pred_layers", 1, 3, step=1),
    }
    return param


@register_hparams('gvp')
def get_gvp_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Get GVP parameters for hyper-parameter tuning.
    """
    
    param = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, step=16),
        "num_layers": trial.suggest_int("num_layers", 2, 4, step=1),
        "pred_hidden_dim": trial.suggest_int("pred_hidden_dim", 16, 256, step=16),
        "pred_dropout": trial.suggest_float("pred_dropout", 0.0, 0.5),
        "pred_layers": trial.suggest_int("pred_layers", 1, 3, step=1),
        "normalization_factor": trial.suggest_float("normalization_factor", 10.0, 1000.0, step=10.0),
        "drop_rate": trial.suggest_float("drop_rate", 0.0, 0.5),
    }
    return param