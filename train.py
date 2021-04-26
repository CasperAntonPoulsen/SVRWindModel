from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from joblib import dump, load

import numpy as np
import math
import pandas as pd

from influxdb import InfluxDBClient # install via "pip install influxdb"

import os
import warnings
import sys
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
#mlflow.set_tracking_uri("http://13.79.151.110:5000/")
#mlflow.set_tracking_uri("http://training.itu.dk:5000/")
#mlflow.set_experiment("sklearnSvrWindCaap")
import logging
import datetime

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mae = mean_absolute_error(actual, pred)
	r2 = r2_score(actual, pred)
	return rmse, mae, r2

def get_df(results):
	values = results.raw["series"][0]["values"]
	columns = results.raw["series"][0]["columns"]
	df = pd.DataFrame(values, columns=columns).set_index("time")
	df.index = pd.to_datetime(df.index) # Convert to datetime-index
	return df

class WindVectorTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.directions =  ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
		self.md_dict = {}
		for i in range(len(self.directions)):
			md = 270 - (i*22.5)
			if md < 0:
				md += 360
			self.md_dict[self.directions[i]]=md

	def calculate_u(self, wwd, ws):
		return ws*math.cos(self.md_dict[wwd])

	def calculate_v(self, wwd, ws):
		return ws*math.sin(self.md_dict[wwd])

	def transform(self, X, y = None):
		X_ = X
		rows = []
		for index, row in X_.iterrows():
			vector = [self.calculate_u(row["Direction"],row["Speed"]),
					  self.calculate_v(row["Direction"],row["Speed"])]
			rows.append(pd.DataFrame.from_dict({index:vector},columns=["u","v"],orient="index"))

		return pd.concat(rows)

	def fit(self, X, y = None):
		return self


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	np.random.seed(40)

	if sys.argv[4] == "now":
		now = datetime.datetime.utcnow().strftime("'%Y-%m-%dT%H:%M:%SZ'")
	elif len(sys.argv) > 4:
		now = sys.argv[4]
	else:
		now = datetime.datetime.utcnow().strftime("'%Y-%m-%dT%H:%M:%SZ'")

	daysDelta = sys.argv[5] if len(sys.argv) > 5 else "90"
	try:

		client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
		client.switch_database('orkney')


		# Get the last 90 days of power generation data
		generation = client.query(
			"SELECT * FROM Generation where time > {}-{}d".format(now,daysDelta)
			) # Query written in InfluxQL

		# Get the last 90 days of weather forecasts with the shortest lead time
		wind  = client.query(
			"SELECT * FROM MetForecasts where time > {}-{}d and time <= {} and Lead_hours = '1'".format(now,daysDelta,now)
			) # Query written in InfluxQL

	except Exception as e:
		logger.exception(
			"Unable to download training & test data, check your internet connection. Error: %s", e
		)

	gen_df = get_df(generation)
	wind_df = get_df(wind)

	gen_df_alligned = pd.merge_asof(wind_df,gen_df,left_index=True, right_index=True)[["Total"]]

	train_length = int(len(gen_df_alligned)*0.9)

	train_X = wind_df.iloc[:train_length]
	test_X = wind_df.iloc[train_length:]

	train_y = gen_df_alligned.iloc[:train_length]
	test_y = gen_df_alligned.iloc[train_length:]

	gamma = [float(i) for i in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.1]
	kernel = sys.argv[2].split(",") if len(sys.argv) > 2 else ["rbf"]
	C = [float(i) for i in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1.0]

	with mlflow.start_run():

		pipeline = Pipeline(steps=[
			("WindVector_transform",WindVectorTransformer()),
			("svm_model", svm.SVR())
		])
		parameters = {'svm_model__kernel':kernel,
              'svm_model__C':C,
              'svm_model__gamma':gamma}

		tscv = TimeSeriesSplit(n_splits=5)
		pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=15, cv= tscv)

		pipeline.fit(train_X, np.ravel(train_y))

		bestParams = pipeline.best_params_

		predicted_qualities = pipeline.predict(test_X)

		(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

		print("SVR model (gamma={}, kernel={}, C={})".format(gamma, kernel, C))
		print("  RMSE: %s" % rmse)
		print("  MAE: %s" % mae)
		print("  R2: %s" % r2)

		mlflow.log_param("bestGamma",bestParams['svm_model__gamma'])
		mlflow.log_param("bestKernel",bestParams['svm_model__kernel'])
		mlflow.log_param("bestC",bestParams['svm_model__C'])
		mlflow.log_param("dateTimeUTC",now)
		mlflow.log_metric("rmse", rmse)
		mlflow.log_metric("r2", r2)
		mlflow.log_metric("mae", mae)

		tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

		if tracking_url_type_store != "file":
			mlflow.sklearn.save_model(pipeline, "model")
		else:
			mlflow.sklearn.log_model(pipeline, "model")
