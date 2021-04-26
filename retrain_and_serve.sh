
export MLFLOW_TRACKING_URI="http://training.itu.dk:5000/"
git pull
rm -r model
mlflow run .
killall -SIGKILL mlflow
mlflow models serve -m model
