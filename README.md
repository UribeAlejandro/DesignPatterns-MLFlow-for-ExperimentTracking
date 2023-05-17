# Speed up Training with MLFlow and Strategy Pattern

## Data

The `Higgs` dataset is a dataset of proton-proton collision events collected at the Large Hadron Collider (`LHC`). The dataset contains information about the properties of the particles produced in the collisions, such as their momentum, energy, and charge. The `Higgs` dataset is used to study the properties of the `Higgs` boson, a fundamental particle that was discovered at the LHC in 2012. The `Higgs` boson is a key component of the Standard Model of particle physics, and its discovery was a major breakthrough in physics.

The `Higgs` dataset is a valuable resource for physicists who are studying the `Higgs` boson and its properties. The dataset  is large and complex, but it can be used to learn a great deal about the `Higgs` boson. The `Higgs` dataset has been used to make important discoveries about the `Higgs` boson, and it will continue to be used to study the `Higgs` boson in the years to come.

Here are some of the key features of the `Higgs` dataset:

- The dataset contains information about the properties of millions of proton-proton collision events.
- The dataset is large and complex, making it difficult to analyze.
- The dataset has been used to make important discoveries about the `Higgs` boson.
- The dataset will continue to be used to study the `Higgs` boson in the years to come.

## MLFlow and Strategy Pattern for Machine Learning Experimentation
This project demonstrates how to use `MLFlow` and the `Strategy pattern` to speed up machine learning experimentation.

### MLFlow
`MLFlow` is an open-source platform for managing the ML lifecycle, including experiment tracking, model packaging, and model serving. `MLFlow` can be used to track the progress of machine learning experiments, store and share models, and deploy models to production.

### `Strategy Pattern`
The `Strategy pattern` is a design pattern that allows you to decouple the algorithm from the context in which it is used. This can be useful for machine learning experimentation, as it allows you to easily experiment with different algorithms without having to modify the code that loads and prepares the data.

Benefits of Using `MLFlow` and the `Strategy Pattern`
There are several benefits to using `MLFlow` and the `Strategy pattern` for machine learning experimentation:

`MLFlow` provides a centralized platform for tracking experiments, storing models, and deploying models. This can help you to keep track of your experiments, share your models with others, and deploy your models to production.

On the other hand, the `Strategy pattern` allows you to easily experiment with different algorithms without having to modify the code that loads and prepares the data. This can save you time and effort when experimenting with different machine learning algorithms.

### Requirements
To use this project, you will need to have `Python 3.9` or later, and install the following dependencies.

```bash
pip install -r requirements_dev.txt
```

Once you have installed the necessary dependencies, you can run the following command to start a `MLFlow` server:

```bash
export MLFLOW_PORT=5000 &&
export MLFLOW_ARTIFACT_ROOT=mlruns
export MLFLOW_TRACKING_URI=sqlite:///database/mlruns.db &&
mlflow server --backend-store-uri=$MLFLOW_TRACKING_URI --default-artifact-root=file:$MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port $MLFLOW_PORT
```

Modify `data/config/config.yaml` according to your needs. Then, you can start a training loop as follows:

```bash
python -m src.main [commands]
```

The `commands` are listed below:

```bash
usage: [-h] [-f FILEPATH] [-d DATASET] [-l LOCATION] [-e EXPERIMENT_NAME]

options:
  -h, --help            show this help message and exit
  -f FILEPATH, --filepath FILEPATH
                        Path to configuration file
  -d DATASET, --dataset DATASET
                        Dataset name
  -l LOCATION, --location LOCATION
                        Folder to store data
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        MLFlow's experiment name
```

The project will then load the Higgs dataset, train a model, and track the experiment using `MLFlow`. You can then view the results of the experiment in the `MLFlow UI` as follows:

```bash
mlflow ui
```

## Conclusion
This project demonstrates how to use `MLFlow` and the `Strategy pattern` to speed up machine learning experimentation. `MLFlow` and the `Strategy pattern` can help you to keep track of your experiments, share your models with others, and deploy your models to production.
