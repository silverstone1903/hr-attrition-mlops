from sklearn.ensemble import RandomForestClassifier
from ml.utils import scorer
import pandas as pd

def trainer(x_train, y_train, x_test, y_test, cols, params, seed = 2021):

    model = RandomForestClassifier(**params)

    model.fit(x_train, y_train)
    preds = model.predict_proba(x_test)[:, 1]

    feature_df = pd.DataFrame(list(zip(cols, model.feature_importances_)),
                              columns=["feature", "importance"])
    feature_df = feature_df.sort_values(
        by='importance',
        ascending=False)

    return model, preds, feature_df