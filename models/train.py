import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import mlflow
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self):
        self.data = pd.read_csv('../Data/train/DataTrain.csv')
        self.target = pd.read_csv('../Data/train/TargetTrain.csv')
        self.lgbm_pounds = {"n_estimators": (50, 350), "boosting_type": (0, 2),
                            "num_leaves": (2, 100), "class_weight": (2, 100),
                            "learning_rate": (-4, 0), "subsample_for_bin": (1, 8),
                            "min_split_gain": (0.0001, 1), "min_child_weight": (-5, -1),
                            "min_child_samples": (5, 100), "subsample": (-5, 0)
                            }
        self.xgb_pounds = {"max_depth": (1, 20), "learning_rate": (0, 1),
                           "booster": (0, 1), "gamma": (0, 100),
                           "min_child_weight": (0, 100), "max_delta_step": (0, 100),
                           "subsample": (0, 1), "colsample_bytree": (0, 1),
                           "colsample_bylevel": (0, 1), "colsample_bynode": (0, 1),
                           "reg_alpha": (0, 10), "reg_lambda": (0, 10),
                           "scale_pos_weight": (1, 25)
                           }
        self.cat_pounds = {"learning_rate": (0.00001, 1), "depth": (3, 10),
                           "l2_leaf_reg": (0, 30),
                           "grow_policy": (0, 2), "min_data_in_leaf": (1, 20),
                           "bagging_temperature": (0, 10), "random_strength": (1, 20)
                           }
        self.best_model = []

    def train_encoder(self):
        lb = LabelEncoder()
        lb.fit(self.target)
        self.target = pd.Series(lb.transform(self.target)).replace({0: 1, 1: 0})

    def train_model(self):
        optimizer_lgbm = BayesianOptimization(
            f=self.lgbm_cv,
            pbounds=self.lgbm_pounds,
            random_state=1234,
            verbose=2
        )

        optimizer_xgb = BayesianOptimization(
            f=self.xgb_cv,
            pbounds=self.xgb_pounds,
            random_state=1234,
            verbose=2
        )
        #
        optimizer_cat = BayesianOptimization(
            f=self.cat_cv,
            pbounds=self.cat_pounds,
            random_state=1234,
            verbose=2
        )
        mlflow.set_experiment(experiment_name='predict In_out/AM')

        optimizer_xgb.maximize(n_iter=500)
        optimizer_lgbm.maximize(n_iter=500)
        optimizer_cat.maximize(n_iter=500)

    def lgbm(self, n_estimators, boosting_type, num_leaves, class_weight, learning_rate, subsample_for_bin,
             min_split_gain, min_child_weight, min_child_samples, subsample):
        estimator = LGBMClassifier(
            n_estimators=int(n_estimators),
            boosting_type=['gbdt', 'dart', 'goss'][int(round(boosting_type))],
            num_leaves=int(num_leaves),
            class_weight={0: 1, 1: int(class_weight)},
            learning_rate=10 ** learning_rate,
            subsample_for_bin=10 ** int(subsample_for_bin),
            min_split_gain=int(min_split_gain),
            min_child_weight=int(10 ** min_child_weight),
            min_child_samples=int(min_child_samples),
            subsample=10 ** subsample,
            random_state=2
        )
        return estimator

    def xgb(self, max_depth, learning_rate, booster, gamma, min_child_weight, max_delta_step,
            subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha,
            reg_lambda, scale_pos_weight):
        estimator = XGBClassifier(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            booster=['gbtree', 'dart'][int(round(booster))],
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=2
        )
        return estimator

    def cat(self, learning_rate, depth, l2_leaf_reg, grow_policy, min_data_in_leaf, bagging_temperature,
            random_strength):
        estimator = CatBoostClassifier(
            iterations=200,
            learning_rate=learning_rate,
            depth=int(depth),
            l2_leaf_reg=l2_leaf_reg,
            random_seed=32,
            grow_policy=['SymmetricTree', 'Depthwise', 'Lossguide'][int(round(grow_policy))],
            min_data_in_leaf=int(min_data_in_leaf),
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            verbose=False,
            task_type='GPU'
        )
        return estimator

    def lgbm_cv(self, n_estimators, boosting_type, num_leaves, class_weight, learning_rate, subsample_for_bin,
                min_split_gain, min_child_weight, min_child_samples, subsample):
        estim = self.lgbm(n_estimators, boosting_type, num_leaves, class_weight, learning_rate, subsample_for_bin,
                          min_split_gain, min_child_weight, min_child_samples, subsample)
        cval = cross_val_score(estim, self.data, self.target,
                               scoring=make_scorer(f1_score), cv=5)

        with mlflow.start_run(run_name='lightgbm'):
            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('boosting_type', boosting_type)
            mlflow.log_param('num_leaves', num_leaves)
            mlflow.log_param('class_weight', class_weight)
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('subsample_for_bin', subsample_for_bin)
            mlflow.log_param('min_split_gain', min_split_gain)
            mlflow.log_param('min_child_weight', min_child_weight)
            mlflow.log_param('min_child_samples', min_child_samples)
            mlflow.log_param('subsample', subsample)
            mlflow.log_metric("f1_score", cval.mean())

        return cval.mean()

    def xgb_cv(self, max_depth, learning_rate, booster, gamma, min_child_weight, max_delta_step,
               subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha, reg_lambda,
               scale_pos_weight):
        estim = self.xgb(max_depth, learning_rate, booster, gamma, min_child_weight, max_delta_step,
                         subsample, colsample_bytree, colsample_bylevel, colsample_bynode,
                         reg_alpha, reg_lambda, scale_pos_weight)
        cval = cross_val_score(estim, self.data, self.target,
                               scoring=make_scorer(f1_score), cv=5)
        with mlflow.start_run(run_name='xgboost'):
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('booster', booster)
            mlflow.log_param('gamma', gamma)
            mlflow.log_param('min_child_weight', min_child_weight)
            mlflow.log_param('max_delta_step', max_delta_step)
            mlflow.log_param('subsample', subsample)
            mlflow.log_param('colsample_bytree', colsample_bytree)
            mlflow.log_param('colsample_bylevel', colsample_bylevel)
            mlflow.log_param('colsample_bynode', colsample_bynode)
            mlflow.log_param('reg_alpha', reg_alpha)
            mlflow.log_param('reg_lambda', reg_lambda)
            mlflow.log_param('scale_pos_weight', scale_pos_weight)
            mlflow.log_metric("f1_score", cval.mean())

        return cval.mean()

    def cat_cv(self, learning_rate, depth, l2_leaf_reg, grow_policy, min_data_in_leaf,
               bagging_temperature, random_strength):
        estim = self.cat(learning_rate, depth, l2_leaf_reg, grow_policy, min_data_in_leaf,
                         bagging_temperature, random_strength)
        cval = cross_val_score(estim, self.data, self.target,
                               scoring=make_scorer(f1_score), cv=5)
        with mlflow.start_run(run_name='catboost'):
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('depth', depth)
            mlflow.log_param('l2_leaf_reg', l2_leaf_reg)
            mlflow.log_param('grow_policy', grow_policy)
            mlflow.log_param('min_data_in_leaf', min_data_in_leaf)
            mlflow.log_param('bagging_temperature', bagging_temperature)
            mlflow.log_param('random_strength', random_strength)
            mlflow.log_metric("f1_score", cval.mean())

        return cval.mean()


if __name__ == '__main__':
    models = Trainer()
    models.train_encoder()
    models.train_model()
