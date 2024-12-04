from sklearn.ensemble import AdaBoostClassifier
from dpdt import DPDTreeClassifier
import numpy as np

class AdaBoostDPDT(AdaBoostClassifier):
    def __init__(
            self,
            config_base=dict(
                max_depth=3, 
                min_samples_split=2, 
                min_impurity_decrease=0, 
                cart_nodes_list=(32,), 
                min_samples_leaf=1, 
                min_weight_fraction_leaf=0,
                max_features=None,),
            random_state=None, 
            n_estimators=50,
            learning_rate=1,
            algorithm="SAMME",
        ):
        super().__init__(
            estimator=DPDTreeClassifier(
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
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__config_base": {
        "values":[
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [32, 16], 'min_samples_leaf': 12.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [32, 16], 'min_samples_leaf': 12.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 12.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 12.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 12.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 12.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 12.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 19.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 19.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [32], 'min_samples_leaf': 15.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8], 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [32], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8], 'min_samples_leaf': 10.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 10.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [32], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 8], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [32, 16], 'min_samples_leaf': 18.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 8], 'min_samples_leaf': 19.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8], 'min_samples_leaf': 24.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 23.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 8], 'min_samples_leaf': 21.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 42.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 4, 4], 'min_samples_leaf': 19.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 4, 4], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 4, 4], 'min_samples_leaf': 2.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 4, 4], 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [4, 4, 4], 'min_samples_leaf': 34.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'cart_nodes_list': [8, 4], 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.05, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 10000, 'cart_nodes_list': [16, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 3.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [8, 4], 'min_samples_leaf': 20.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [8, 4], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [8, 4], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [8, 4], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 31.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 16.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 46.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 17.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 14.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 9.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 9.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 23.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 'log2', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 24.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 3.0, 'max_features': 10000, 'cart_nodes_list': [8], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 3.0, 'max_features': 10000, 'cart_nodes_list': [32], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [32], 'min_samples_leaf': 46.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 47.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 3.0, 'max_features': 'log2', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 14.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 13.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 16.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 11.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 8.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [8], 'min_samples_leaf': 4.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 3.0, 'max_features': 'log2', 'cart_nodes_list': [32, 16], 'min_samples_leaf': 5.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 4.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 3.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 31.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
{'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [32], 'min_samples_leaf': 30.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,],
        "probabilities": [0.010209571347986612,0.010209571347986612,0.011433314407554889,0.012006550871611793,0.012006550871611793,0.012006550871611793,0.012006550871611793,0.01067111399556385,0.01067111399556385,0.011152975143486123,0.009699221316200808,0.010856587571856518,0.009719414309098454,0.00980796513955205,0.010291093825455866,0.009719819472679378,0.009719819472679378,0.010291093825455866,0.010461226514645735,0.009669398582570645,0.009423515245078805,0.009818600852956378,0.010720234302254938,0.010176740708228372,0.009818600852956378,0.00972263801513721,0.009455417590980947,0.009964228702385893,0.00949819326397312,0.010052751407442237,0.010139787299778083,0.009753401111817593,0.010087348407542978,0.0097765919655742,0.009416758333627677,0.010331996886532647,0.00967027534782249,0.009738868523834718,0.010207051523135694,0.010591940669098123,0.009532538867553262,0.01047171008449587,0.009354197669279257,0.009962945413261,0.009962945413261,0.011072166440498903,0.010306989999475422,0.01105817435772133,0.011113432681291096,0.009572918105284762,0.009737003934458704,0.009047976060210738,0.00956032086669383,0.009086686687588873,0.009376479606217764,0.00945493772688823,0.00942408284417643,0.01050359942646942,0.010207530128814079,0.010484009671238095,0.009831764492618347,0.009911806727169948,0.00959360736782326,0.009251571934465935,0.009500849410076805,0.009274403202231827,0.009354061481795418,0.009384716330628165,0.009961568135300592,0.009404840380732306,0.009376479606217764,0.00963940597785278,0.00982683650512178,0.009572918105284762,0.011115670092151224,0.009937816420210093,0.009715652981142495,0.010226899771333458,0.009626718729873616,0.010759933473254045,0.00959360736782326,0.009337730111598809,0.00942720984123571,0.009422097207982547,0.010546595124399257,0.010207530128814079,0.010004758757360815,0.009372692778078895,0.00945493772688823,0.00949144107043085,0.009242983279701298,0.009435013585690153,0.00909758498364794,0.009422097207982547,0.009354061481795418,0.010935274984659836,0.009657420237833496,0.010759933473254045,0.00977625258847242,0.009928166187245723]    },

    "model__random_state": {
        "values": [0, 1, 2, 3, 4],
        "probabilities": [1/5] * 5
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
        "value": "adaboost_dpdt_c"
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "adaboost_dpdt_c"
    },
})
