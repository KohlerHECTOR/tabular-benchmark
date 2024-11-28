import numpy as np

config_random = {
    "model_type": {
        "value": "sklearn-tree"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": {
        "value": 5
    },

    "model__cost_complexity": {
        "value": 0
    },

    "model__time_limit": {
        "value": 1800
    },

    "model__max_num_nodes": {
        "values": [3, 5, 7, 11, 17, 25, 31],
        "probabilities": [1/7]*7
    },

    "model__min_samples_leaf": {  # Not exactly like Hyperopt
        "distribution": "q_log_uniform_values",
        "min": 1.5,
        "max": 50.5,
        "q": 1
    },

    "model__n_thresholds": {
        "values": [5, 10, 20, 50],
        "probabilities": [0.25]*4
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
        "value": "sklearn-tree"
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
        "value": "pystreed_r"
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "pystreed_r"
    },
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "pystreed_c"
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "pystreed_c"
    },
})
