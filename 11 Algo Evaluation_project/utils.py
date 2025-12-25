from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd


def get_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include="number").columns
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
    )
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    pre.fit(X)
    return pre


def get_models(random_state=42):
    models = [
        LogisticRegression(random_state=random_state),
        DecisionTreeClassifier(max_depth=3, random_state=random_state),
        RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=random_state
        ),
        HistGradientBoostingClassifier(max_depth=3, random_state=random_state),
        XGBClassifier(n_estimators=100, max_depth=3, random_state=random_state),
    ]
    return models


def single_model_eval(model, xtrain, ytrain, xtest, ytest):
    cv_scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    cv_avg = cv_scores.mean()
    model.fit(xtrain, ytrain)
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    f1_train = f1_score(ytrain, ypred_train)
    f1_test = f1_score(ytest, ypred_test)
    res = {
        "model": model,
        "name": type(model).__name__,
        "f1_cv": cv_avg.round(4),
        "f1_train": round(f1_train, 4),
        "f1_test": round(f1_test, 4),
    }
    return res


def algo_evaluation(models: list, xtrain, ytrain, xtest, ytest):
    results = []
    for i in models:
        r = single_model_eval(i, xtrain, ytrain, xtest, ytest)
        print(r)
        results.append(r)
        print("==================================================")
    # Show results in dataframe
    res_df = pd.DataFrame(results)
    # Select the best model from above
    sort_df = res_df.sort_values(by=["f1_cv"], ascending=False).reset_index(drop=True)
    best_model = sort_df.loc[0, "model"]
    return sort_df, best_model
