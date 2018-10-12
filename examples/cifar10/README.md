# Cifar10 example project

In this folder we provide an example of how to use the package
on Cifar10 data.


Set `MLFLOW_TRACKING_URI` environment variable to setup output folder.
```
export MLFLOW_TRACKING_URI="/path/to/local/project/output/folder"
```


## Run training


## Run inference


## Visualize logs

```
mlflow ui -h 0.0.0.0 -p 6006 --file-store $MLFLOW_TRACKING_URI
```