# Detoxification Task

# Authors:

**Kirilin Anton - B20-RO** 

a.kirilin@innopolis.university

## Description 📔

In this project you can find .ipynb notebooks with code for training different versions of T5 model for text detoxification. There are python script to run already fine-tuned models, load the datasets, prepare visualising materials. There are also reports of research work done during the assignment

## How to start 🚀

In order to run all the script you have to do the following procedure:

1.  Clone this git repo to your local machine

```bash
git clone https://github.com/Lumiwarum/detoxification/tree/main
```

1. Create a new python virtual environment and activate it

```bash
python3 -m venv /env
source /env/bin/activate
```

1. Install all the requrements

```bash
pip install -r requirements.txt
```

Now you’re ready to run all python script in this repo.

## Structure  📦

```
detoxification
├── README.md # The top-level README
│
├── data
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data to be stored
│
├── models       #  link to models checkpoints
│
├── notebooks    #  Jupyter notebooks.
│
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization   # Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

The structure was taken from [here](https://www.notion.so/Task-description-588e0cb88be4416ea7311426b1d9b360?pvs=21)

[Notebooks](https://www.notion.so/notebooks/) folder contains all the .ipynb file that I used during the research process

[The report](https://www.notion.so/Solution-Building-33f0b7d8aced43118f38d369a50fd137?pvs=21) about baseline and how I have approached the problem

[The report](https://www.notion.so/Final-Solution-a493a971f44a4ed8a85d203a366e7880?pvs=21) about my final model and overall evaluation is here

## Usage:

### Data processing:

1. [src/data/download_dataset.py](https://www.notion.so/Detoxification-Task-f978e900e9784df78e1baf34a3d94af8?pvs=21) - downloads the dataset and extracts it into `data/raw` directory.
2. [src/data/make_datasets.py](https://www.notion.so/Detoxification-Task-f978e900e9784df78e1baf34a3d94af8?pvs=21) - uses the download_dataset.py script and create a cropped processed dataset described in the reports

### Train model:

[src/models/train_model.py](https://www.notion.so/Detoxification-Task-f978e900e9784df78e1baf34a3d94af8?pvs=21) - fine-tunes the t5 from Skolkovo on the dataset made by make_dataset.py

### Predict:

[src/models/predict_model.py](https://www.notion.so/Task-description-588e0cb88be4416ea7311426b1d9b360?pvs=21) - allows you to run the model for detoxifying your sentence. If the model is not fine-tuned - the default model will be from the Skolkovo versio

[Solution Building](https://www.notion.so/Solution-Building-33f0b7d8aced43118f38d369a50fd137?pvs=21)

[Task description](https://www.notion.so/Task-description-588e0cb88be4416ea7311426b1d9b360?pvs=21)

[Final Solution](https://www.notion.so/Final-Solution-a493a971f44a4ed8a85d203a366e7880?pvs=21)