killall -g mlflow
git pull
rm -r model
mlflow run .
mlflow models serve -m model
