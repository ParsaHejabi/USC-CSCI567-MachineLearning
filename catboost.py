import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# set data directory

# load data
X = pd.read_csv("./data/train_values.csv", index_col="building_id")
y = pd.read_csv("./data/train_labels.csv", index_col="building_id")
X_test = pd.read_csv("./data/test_values.csv", index_col="building_id")

# set categorical columns
cat_cols = [
    "geo_level_1_id",
    "geo_level_2_id",
    "geo_level_3_id",
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
    "count_floors_pre_eq",
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
    "legal_ownership_status",
    "has_secondary_use",
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other",
]

# set CatBoost hyperparameters
best_params = {
    "random_seed": 42,
    "use_best_model": True,
    "bagging_temperature": 0.00824,
    "boosting_type": "Ordered",
    "border_count": 11,
    "colsample_bylevel": 0.3053,
    "depth": 9,
    "l2_leaf_reg": 7,
    "learning_rate": 0.065,
    "min_data_in_leaf": 26,
    "n_estimators": 7537,
    "od_type": "IncToDec",
    "random_strength": 0.00044,
    "eval_metric": "TotalF1",
}


# split train and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

# train the CatBoost model on the training data
model = CatBoostClassifier(cat_features=cat_cols, **best_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=1000)

# make predictions on the validation data
preds = model.predict(X_val)

# calculate accuracy using F1 score
accuracy = f1_score(y_val, preds, average="micro")
print("Accuracy on validation set:", accuracy)

# train the CatBoost model using K-fold cross-validation
kf = KFold(n_splits=5)
best_models = []
val_acc = 0

for train_index, val_index in kf.split(X):
    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index, :]

    X_val = X.iloc


model.save_model("catboost_model.bin")
