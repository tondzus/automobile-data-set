import os
import sys
import requests
import argparse

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


def prepare_data(path):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/' \
          'imports-85.data'
    header = 'symboling,normalized-losses,make,fuel-type,aspiration,' \
             'num-of-doors,body-style,drive-wheels,engine-location,' \
             'wheel-base,length,width,height,curb-weight,engine-type,' \
             'num-of-cylinders,engine-size,fuel-system,bore,stroke,' \
             'compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price'
    res = requests.get(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fp:
        fp.write(header)
        fp.write('\n')
        fp.write(res.text)


def preprocess_dataset(df: pd.DataFrame):
    df = df.dropna()
    ohe_encoder = ce.OneHotEncoder(impute_missing=False,
                                   handle_unknown='ignore')
    df = ohe_encoder.fit_transform(df)

    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df


def load_dataset(path):
    autos = pd.read_csv(path, na_values=['?'])
    autos.drop(['symboling', 'normalized-losses'], inplace=True, axis=1)
    autos = preprocess_dataset(autos)
    return autos.drop('price', axis=True), autos['price']


def autos_feature_selection(X, y):
    model = Lasso(alpha=0.01, fit_intercept=False)
    model.fit(X, y)

    feature_selector = SelectFromModel(model, prefit=True)
    dropped_columns = pd.concat(
        [pd.Series(X.columns), pd.Series(feature_selector.get_support())],
        axis=1)
    to_stay_columns = dropped_columns[dropped_columns[1] == True]
    to_stay = to_stay_columns[0].str.split('_').apply(lambda l: l[0]).unique()

    all_features = dropped_columns[0].str.split('_').apply(lambda l: l[0]).unique()
    to_go = set(all_features) - set(to_stay)
    return list(to_stay), list(to_go)


def process_features(features_to_stay, *dfs):
    new_dfs = []
    for df in dfs:
        columns = [col for col in df.columns
                   if all(not col.startswith(f) for f in features_to_stay)]
        new_dfs.append(df.drop(columns, axis=1))

    return new_dfs


def autos_test_price_prediction(X_train, y_train, X_test, y_test):
    #best of 5
    best_score = sys.maxsize
    for _ in range(10):
        regressor = MLPRegressor((100, ), early_stopping=True, max_iter=5000,
                                 validation_fraction=0.2, activation='logistic')
        regressor.fit(X_train, y_train)
        score = mean_squared_error(regressor.predict(X_test), y_test)
        if score < best_score:
            best_score = score
    return best_score


def main():
    abs_path = lambda path: os.path.abspath(os.path.expanduser(path))
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=abs_path,
                        default=abs_path('data/autos.csv'),
                        help='Path to auto1 dataset with header')
    parser.add_argument('--random', action='store_true',
                        help='If not set results will be the same on every run')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        print('Dataset already present - no download')
    else:
        print('Dataset not present - downloading')
        prepare_data(args.path)

    if not args.random:
        np.random.seed(0)

    X, y = load_dataset(args.path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    features_to_stay, features_to_go = autos_feature_selection(X_train, y_train)
    print('Keeping features:', sorted(features_to_stay))
    print('Droping features:', sorted(features_to_go))

    all_features = autos_test_price_prediction(X_train, y_train, X_test, y_test)
    print('Dataset with %s columns (MSE): %s' %
          (X_train.shape[1], all_features))

    X_train_selected, X_test_selected = process_features(
        features_to_stay, X_train, X_test
    )
    selected_features = autos_test_price_prediction(X_train_selected, y_train,
                                                    X_test_selected, y_test)
    print('Dataset with %s columns (MSE): %s' %
          (X_train_selected.shape[1], selected_features))


if __name__ == '__main__':
    main()
