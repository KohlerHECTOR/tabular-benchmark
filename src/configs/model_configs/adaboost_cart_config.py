from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoostCART(AdaBoostClassifier):
    def __init__(
            self,
            max_depth=1, 
            min_samples_split=2, 
            min_impurity_decrease=0, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0,
            max_leaf_nodes=None,
            max_features=None,
            random_state=None, 
            n_estimators=50,
            learning_rate=1,
            algorithm="SAMME",
        ):
        super().__init__(
            estimator=DecisionTreeClassifier(
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_impurity_decrease=min_impurity_decrease, 
                min_samples_leaf=min_samples_leaf, 
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state,
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )

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
    "model__max_depth": {
        "values": [2, 3],
        "probabilities": [0.4, 0.6]
    },
    
    "model__random_state": {
        "values": [0, 1, 2, 3, 4],
        "probabilities": [1/5] * 5
    },

    "model__min_samples_split": {
        "values": [2, 3],
        "probabilities": [0.95, 0.05]
    },

    "model__min_impurity_decrease": {
        "values": [0.0, 0.01, 0.02, 0.05],
        "probabilities": [0.85, 0.05, 0.05, 0.05],
    },

    "model__max_leaf_nodes": {
        "values": [2**3, 5, 7],
        "probabilities": [0.85, 0.05, 0.1]
    },

    "model__min_samples_leaf": {  # Not exactly like Hyperopt
        "distribution": "q_log_uniform_values",
        "min": 1.5,
        "max": 50.5,
        "q": 1
    },

    "model__min_weight_fraction_leaf": {
            "values": [0.0, 0.01],
            "probabilities": [0.95, 0.05]
        },
    
    "model__max_features": {
        "values": ["sqrt", "log2", 10_000],
        "probabilities": [0.5, 0.25, 0.25]
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
