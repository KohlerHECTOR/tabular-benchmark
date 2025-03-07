import sys
sys.path.append("src")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dpdt import DPDTreeClassifier, DPDTreeRegressor
from pystreed import STreeDClassifier, STreeDRegressor
from xgboost import XGBClassifier, XGBRegressor
from models.skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch
from models.skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch, create_rtdl_mlp_regressor_skorch
from models.TabSurvey.models.saint import SAINT



total_config = {}
model_keyword_dic = {}

## ADD YOU MODEL HERE ##
from configs.model_configs.cart_config import config_classif, config_regression, config_classif_default, config_regression_default #replace template.py by your parameters
keyword = "cart"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = DecisionTreeRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = DecisionTreeClassifier
#############################


from configs.model_configs.dpdt_config import config_classif, config_regression, config_classif_default, config_regression_default #replace template.py by your parameters
keyword = "dpdt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = DPDTreeRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = DPDTreeClassifier


from configs.model_configs.pystreed_config import config_classif, config_regression, config_classif_default, config_regression_default #replace template.py by your parameters
keyword = "pystreed"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = STreeDRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = STreeDClassifier


from configs.model_configs.gpt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "gbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = GradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = GradientBoostingClassifier

from configs.model_configs.adaboost_cart_config import config_classif, config_classif_default, AdaBoostCART
keyword = "boost_cart"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
}
model_keyword_dic[config_classif["model_name"]["value"]] = AdaBoostCART


from configs.model_configs.adaboost_dpdt_config import config_classif, config_classif_default, AdaBoostDPDT
keyword = "boost_dpdt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
}
model_keyword_dic[config_classif["model_name"]["value"]] = AdaBoostDPDT


from configs.model_configs.rf_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "rf"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = RandomForestRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = RandomForestClassifier

from configs.model_configs.hgbt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "hgbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = HistGradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = HistGradientBoostingClassifier

from configs.model_configs.xgb_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = XGBClassifier

from configs.model_configs.xgb_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = XGBClassifier

from configs.model_configs.mlp_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "mlp"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_rtdl_mlp_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_rtdl_mlp_skorch

from configs.model_configs.resnet_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "resnet"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_resnet_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_resnet_skorch

from configs.model_configs.ft_transformer_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "ft_transformer"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_ft_transformer_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_ft_transformer_skorch

from configs.model_configs.saint_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "saint"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = SAINT
model_keyword_dic[config_classif["model_name"]["value"]] = SAINT


