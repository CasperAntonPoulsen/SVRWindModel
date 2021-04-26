
export MLFLOW_TRACKING_URI="http://training.itu.dk:5000/"
git pull
rm -r model
mlflow run . --experiment-name sklearnSvrWindCaap
killall -g mlflow
mlflow models serve -m model
