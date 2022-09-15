# from crypt import methods
import sys
import os
import shutil
import time
import traceback
from xml.etree.ElementTree import tostringlist
import flask
from flask import Flask, request, jsonify, render_template, session, redirect
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from flask_cors import CORS, cross_origin

import numpy as np
app = Flask(__name__,template_folder='template')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# se definen los inputs
training_data = 'data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = include[-1]

# Inputs reto
df_transactions = pd.read_csv('data/transactions.csv').sort_values(['store_nbr', 'date'])
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_holidays = pd.read_csv('data/holidays_events.csv')
df_oil = pd.read_csv('data/oil.csv')
df_submission = pd.read_csv('data/sample_submission.csv')
df_stores = pd.read_csv('data/stores.csv')


# se definen los directorios del modelo
model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# Aqui es donde se define la variable para guardar el modelo
model_columns = None
clf = None

# Este endpoint solo se encarga de usar el modelo ya entrenado para regresar una prediccion
@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'

 # en esta ruta es donde debemos definir todo nuestro modelo y su entrenamiento
@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles
    from sklearn.ensemble import RandomForestClassifier as rf

    df = pd.read_csv(training_data)
    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = rf()
    start = time.time()
    clf.fit(x, y)

    joblib.dump(clf, model_file_name)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
    return return_message

# esto solo es para borrar el modelo en memoria
@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'

# Pruebas
@app.route('/lineal')
def trans():
    #global df_train, df_he, df_test
    df_transactions = pd.read_csv('data/transactions.csv').sort_values(['store_nbr', 'date'])
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_holidays = pd.read_csv('data/holidays_events.csv')
    df_oil = pd.read_csv('data/oil.csv')
    df_submission = pd.read_csv('data/sample_submission.csv')
    df_stores = pd.read_csv('data/stores.csv')

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])

    df_fd = df_train.loc[(df_train["date"].dt.day == 1) & (df_train["date"].dt.month == 1)].index
    df_train = df_train.drop(df_fd)

    df_fd=df_train.loc[(df_train["date"].dt.month == 4) & (df_train["date"].dt.day >= 16) & (df_train["date"].dt.day <= 31) & (df_train["date"].dt.year == 2016)].index
    df_train=df_train.drop(df_fd)

    df_fd=df_train.loc[(df_train["date"].dt.month == 5) & (df_train["date"].dt.day >= 1) & (df_train["date"].dt.day <= 16) & (df_train["date"].dt.year == 2016)].index
    df_train=df_train.drop(df_fd)

    df_dates = df_train.groupby(df_train.date)['sales'].mean().reset_index()
   

    df_dates = df_train.groupby(df_train.date)['sales'].mean().reset_index()

    df_train = pd.merge(df_train, df_transactions, on=['date','store_nbr'])

    # df_he
    df_he = df_holidays.drop(["locale_name", "description", "type"], axis=1)

    df_di=df_he.loc[(df_he["locale"] == 'Regional')].index
    df_he=df_he.drop(df_di)

    df_di=df_he.loc[(df_he["transferred"] == True)].index
    df_he=df_he.drop(df_di)

    df_he = df_he.drop(["transferred"], axis=1)

    df_he = df_he.replace(to_replace="Local", value=1)
    df_he = df_he.replace(to_replace="National", value=1)
    df_he.columns = df_he.columns.str.replace('locale', 'IsHoliday')
    df_he["date"] = pd.to_datetime(df_he['date'])


    df_train['IsHoliday'] = df_he['IsHoliday']
    df_train['IsHoliday'] = df_train['IsHoliday'].fillna(0)

    df_train['year'] = ((df_train["date"].dt.year).astype('int')) - 2000

    # X_store = pd.get_dummies(df_train['store_nbr'])

    df_train['date'] = df_train.date.dt.to_period('D')

    df_train = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()

    df_train['id'] = df_train['id'].astype(object)

    # Modelo
    # average_sales = df_train.groupby('date').mean().sales

    y = df_train.unstack(['store_nbr','family']).loc['2017']

    y = y.fillna(0)

    fourier = CalendarFourier(freq='M', order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )

    X = dp.in_sample()

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)

    X_test = dp.out_of_sample(steps=16)

    X_test.index.name='date'

    y_submit = model.predict(X_test)

    y_submit = pd.DataFrame(y_submit, index=X_test.index,
                       columns=y.columns)

    y_submit = y_submit.stack(['store_nbr', 'family'])
    
    y_submit['id'] = y_submit['id'].astype(int)

    y_submit = y_submit.reset_index()
    y_submit = y_submit.reset_index()
    y_submit = y_submit.reset_index()
    df_test = df_test.reset_index()
    df_test = df_test.reset_index()

    y_submit['level_0'] = y_submit['level_0'].astype(int)
    df_test['level_0'] = df_test['level_0'].astype(int)

    y_submit = pd.merge(y_submit, df_test, on="level_0")
    y_submit = y_submit[['date_x','sales']]
   
    result_list = y_submit["sales"].tolist()



    y_submit["date_x"] = y_submit["date_x"].values.astype('datetime64[ns]')
    #y_submit["date_x"] = y_submit["date_x"].dt.strftime("%m/%d/%Y")

    y_submit = y_submit.groupby(['date_x']).sales.sum()
    #y_submit["date_x"] = y_submit["date_x"].values.astype('datetime64[ns]')
    #y_submit["date_x"] = y_submit["date_x"].dt.strftime("%m/%d/%Y")
    print (y_submit)
    #json_result = y_subm
    import json
    json_result = json.dumps(y_submit.to_json())
    #print(json_result)

 
    resp = flask.Response(json_result)
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers["Access-Control-Allow-Headers"]= "X-Requested-With"
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

   # return render_template('main.html',  tables=[y_submit.to_html(classes='data')], titles=y_submit.columns.values)
###########################################
@app.route('/forest')
def forest():
    #global df_train, df_he, df_test
    df_transactions = pd.read_csv('data/transactions.csv').sort_values(['store_nbr', 'date'])
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_holidays = pd.read_csv('data/holidays_events.csv')
    df_oil = pd.read_csv('data/oil.csv')
    df_submission = pd.read_csv('data/sample_submission.csv')
    df_stores = pd.read_csv('data/stores.csv')

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])

    df_fd = df_train.loc[(df_train["date"].dt.day == 1) & (df_train["date"].dt.month == 1)].index
    df_train = df_train.drop(df_fd)

    df_fd=df_train.loc[(df_train["date"].dt.month == 4) & (df_train["date"].dt.day >= 16) & (df_train["date"].dt.day <= 31) & (df_train["date"].dt.year == 2016)].index
    df_train=df_train.drop(df_fd)

    df_fd=df_train.loc[(df_train["date"].dt.month == 5) & (df_train["date"].dt.day >= 1) & (df_train["date"].dt.day <= 16) & (df_train["date"].dt.year == 2016)].index
    df_train=df_train.drop(df_fd)

    df_dates = df_train.groupby(df_train.date)['sales'].mean().reset_index()
   

    df_dates = df_train.groupby(df_train.date)['sales'].mean().reset_index()

    df_train = pd.merge(df_train, df_transactions, on=['date','store_nbr'])

    # df_he
    df_he = df_holidays.drop(["locale_name", "description", "type"], axis=1)

    df_di=df_he.loc[(df_he["locale"] == 'Regional')].index
    df_he=df_he.drop(df_di)

    df_di=df_he.loc[(df_he["transferred"] == True)].index
    df_he=df_he.drop(df_di)

    df_he = df_he.drop(["transferred"], axis=1)

    df_he = df_he.replace(to_replace="Local", value=1)
    df_he = df_he.replace(to_replace="National", value=1)
    df_he.columns = df_he.columns.str.replace('locale', 'IsHoliday')
    df_he["date"] = pd.to_datetime(df_he['date'])


    df_train['IsHoliday'] = df_he['IsHoliday']
    df_train['IsHoliday'] = df_train['IsHoliday'].fillna(0)

    df_train['year'] = ((df_train["date"].dt.year).astype('int')) - 2000

    # X_store = pd.get_dummies(df_train['store_nbr'])

    df_train['date'] = df_train.date.dt.to_period('D')

    df_train = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()

    df_train['id'] = df_train['id'].astype(object)

    # Modelo
    # average_sales = df_train.groupby('date').mean().sales

    y = df_train.unstack(['store_nbr','family']).loc['2017']

    y = y.fillna(0)

    fourier = CalendarFourier(freq='M', order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )

    X = dp.in_sample()

    model = RandomForestRegressor()
    model.fit(X, y)

    # y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)

    X_test = dp.out_of_sample(steps=16)

    X_test.index.name='date'

    y_submit = model.predict(X_test)

    y_submit = pd.DataFrame(y_submit, index=X_test.index,
                       columns=y.columns)

    y_submit = y_submit.stack(['store_nbr', 'family'])
    
    y_submit['id'] = y_submit['id'].astype(int)

    y_submit = y_submit.reset_index()
    y_submit = y_submit.reset_index()
    y_submit = y_submit.reset_index()
    df_test = df_test.reset_index()
    df_test = df_test.reset_index()

    y_submit['level_0'] = y_submit['level_0'].astype(int)
    df_test['level_0'] = df_test['level_0'].astype(int)

    y_submit = pd.merge(y_submit, df_test, on="level_0")
    y_submit = y_submit[['date_x','sales']]
   
    result_list = y_submit["sales"].tolist()



    y_submit["date_x"] = y_submit["date_x"].values.astype('datetime64[ns]')
    #y_submit["date_x"] = y_submit["date_x"].dt.strftime("%m/%d/%Y")

    y_submit = y_submit.groupby(['date_x']).sales.sum()
    #y_submit["date_x"] = y_submit["date_x"].values.astype('datetime64[ns]')
    #y_submit["date_x"] = y_submit["date_x"].dt.strftime("%m/%d/%Y")
    print (y_submit)
    #json_result = y_subm
    import json
    json_result = json.dumps(y_submit.to_json())
    #print(json_result)

 
    resp = flask.Response(json_result)
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers["Access-Control-Allow-Headers"]= "X-Requested-With"
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp



if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
