from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoostCART(AdaBoostClassifier):
    def __init__(
            self,
            config_base=dict(
                max_depth=None, 
                min_samples_split=2, 
                min_impurity_decrease=0, 
                min_samples_leaf=1, 
                min_weight_fraction_leaf=0,
                max_features=None
            ),
            random_state=None, 
            n_estimators=50,
            learning_rate=1,
            algorithm="SAMME",
        ):
        super().__init__(
            estimator=DecisionTreeClassifier(
                 **config_base,
                random_state=random_state,
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )
        self.config_base = config_base

config_random = {
    "model_type": {
        "value": "sklearn"
    },

    "model__learning_rate": {
        'distribution': "log_normal",
        'mu': float(np.log(0.01)),
        'sigma': float(np.log(10.0)),
    },
    "model__n_estimators": {
    "value": 1_000 # Changed as asked by the reviewer
    # "distribution": "q_log_uniform_values",
    # "min": 10.5,
    # "max": 1000.5,
    # "q": 1
    },

    ########### BASE LEARNER
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__config_base":{
        "values":[
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 13.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 13.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 14.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 14.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 9.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 7.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 10.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 22.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 20.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 21.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 21.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 21.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 22.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 24.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 24.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 38.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 36.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 40.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 30.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 27.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 10.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 10.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.01, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.01, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 26.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.01, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.02, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.01, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 40.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 10.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 50.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 41.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 9.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'min_samples_leaf': 17.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'min_samples_leaf': 26.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 19.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.05, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 3.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 3.0, 'max_features': 10000, 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 3.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 15.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 44.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 10.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 45.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 10.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 18.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'min_samples_leaf': 21.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 9.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 32.0} ,],
        "probabilities":[0.010008297777143786,0.010008297777143786,0.010286774083489787,0.010286774083489787,0.009921860962261676,0.009958052871326091,0.010798295015000466,0.010234256141229215,0.009883688648682323,0.009958052871326091,0.009883688648682323,0.010234256141229215,0.010234256141229215,0.010067141160214971,0.010138607913849726,0.010138607913849726,0.009783004037649513,0.009981336962417698,0.009783004037649513,0.009981336962417698,0.009981336962417698,0.010606478400001944,0.009839593740725664,0.009839593740725664,0.009839593740725664,0.010187997181411201,0.009957694642955681,0.010356303317633231,0.00986888520465377,0.009839593740725664,0.009839593740725664,0.010356303317633231,0.010356303317633231,0.00986888520465377,0.00986888520465377,0.010606478400001944,0.010160591485576515,0.01047511438426408,0.010001814847920315,0.009608827432411845,0.009574583716013904,0.011899526712924656,0.010088677385096886,0.010020207163495835,0.010161603086748568,0.01027516637486397,0.011518394382845316,0.010837782692875293,0.01063083150281472,0.011389806775549526,0.010510826621984095,0.010298353905498743,0.010452260611665798,0.010367125325495734,0.010452260611665798,0.009869151932236989,0.0100989569507889,0.009306552833112583,0.009212371234885951,0.00943619303957928,0.009268764286453643,0.009271215251261315,0.009326655741642206,0.00998988345301162,0.009463333724225146,0.009829638146158443,0.010840284806560832,0.010012036562778758,0.00961371081092898,0.00932279115279833,0.00944618477451283,0.00923619549652706,0.009361683390299034,0.009390039403145707,0.010222084515298623,0.009135579454983247,0.009683192568996675,0.009540011479929313,0.0092998173211696,0.009355411676074822,0.009921860962261676,0.010138607913849726,0.009839593740725664,0.009957694642955681,0.009839593740725664,0.00986888520465377,0.009839593740725664,0.00986888520465377,0.009839593740725664,0.00986888520465377,0.00986888520465377,0.009839593740725664,0.009958052871326091,0.009958052871326091,0.009783004037649513,0.011136542160059578,0.009883688648682323,0.009783004037649513,0.009883688648682323,0.009883688648682323]
    },

    "transformed_target": {
        "values": [False]
    },
    "use_gpu": {
        "value": False
    }
}

config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "use_gpu": {
        "value": False
    }
}

config_classif = dict(config_random, **{
    "model_name": {
        "value": "adaboost_cart_c"
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "adaboost_cart_c"
    },
})
