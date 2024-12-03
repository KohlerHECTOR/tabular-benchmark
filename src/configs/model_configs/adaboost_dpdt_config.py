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
            {'max_depth': 2.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
            {'max_depth': 2.0, 'max_features': 'sqrt', 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
            {'max_depth': 3.0, 'max_features': 'sqrt', 'cart_nodes_list': [4, 8], 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
            {'max_depth': 5.0, 'max_features': 'log2', 'cart_nodes_list': [8], 'min_samples_leaf': 6.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
            {'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [16, 2], 'min_samples_leaf': 7.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
            {'max_depth': 5.0, 'max_features': 'sqrt', 'cart_nodes_list': [8, 2, 2], 'min_samples_leaf': 10.0, 'min_samples_split': 2.0, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0} ,
        ],
        "probabilities": [0.16578549, 0.16538915, 0.16758775, 0.16435956, 0.1719927, 0.16488535]
    },

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
