import numpy as np

config_random = {
    "model_type": {
        "value": "sklearn"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": {
        "values": ["None", 2, 3, 4],
        "probabilities": [0.7, 0.1, 0.1, 0.1]
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
        "values": ["None", 5, 10, 15],
        "probabilities": [0.85, 0.05, 0.05, 0.05]
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
        "values": ["sqrt", "log2", "None"],
        "probabilities": [0.5, 0.25, 0.25]
    },

    "transformed_target": {
        "values": [False, True]
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

config_regression = dict(config_random, **{
    "model_name": {
        "value": "cart_r"
    },
    "model__loss": {
        "values": ["squared_error", "absolute_error"],
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "cart_r"
    },
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "cart_c"
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "cart_c"
    },
})
