# Practical Project for Bachelor Thesis: Image-Based Bee Health Classification Using CNNs and Hybrid Machine Learning Models
**Created by: Márió Palágyi**
-Under the supervision of **Prof.(FH) Dr. Deepak Dhungana**

## Table of Contents
1. Introduction
2. How to run the project yourself
3. Libraries Used
4. How to understand the project
5. Summary

## 1. Introduction
This project features the practical implementation of the concept explained in my Bachelor Thesis. The main goal was to examine how traditional Machine Learning algorithms compare to Deep Learning in classifying healthy and unhealthy bees from images.

The traditional models were chosen Random Forest and Support Vector Machine.
For Deep Learning Convolutional Neural Networks were used.

There was a supporting idea to explore: if it is possible to build a hybrid model to classify bee images, by using CNN as feature extraction and SVM as final classification.

In the following documentation, it will be explained how to run the project, which libraries were used and why, how to understand the project as a whole and finally a summary will also be provided.

## 2. How to run the project

The project setup is split into two main steps:

1. Create a local virtual environment and install the required libraries.
2. Prepare the dataset and create the processed train/validation/test split file.

### 1. Create the Python environment

(After download or using GitHub commands, creating the project folder)

From the project root, run:

```powershell
py setup_env.py
```

What this script does is:

- creates a local `.venv` folder if it does not already exist
- installs all dependencies from `requirements.txt`

Then activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Configure Kaggle dataset

Manual download of dataset is also possible, then this step can be skipped:
just go to dataset link:
[https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images]
And click "Download" on top of the page.
Then click "Download Dataset as .zip" (you will need a registered kaggle account for this)
**If the manual download was chosen:**
Skip this step and go to Step 3.: Download and split the dataset

The dataset is downloaded from Kaggle the first time you prepare the data.

You need one of these authentication methods before running the split step:

- set `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables
- or place `kaggle.json` in `%USERPROFILE%/.kaggle/`

Dataset source:
[https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images](https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images)

### 3. Download and split the dataset

Run:

```powershell
py split.py
```

If you get an error at this step, make sure your data directory is the following (the bee_imgs folder might have a nester folder also called "bee_images" inside, but it should only have the raw .pngs):
├── data/
│   ├── bee_imgs/
│   ├── processed/
│   │   └── processed_bee_data_20260330_151740.csv
│   └── raw/
│       └── bee_data.csv

By default, `split.py` will:

- check if `data/raw/bee_data.csv` and the bee images already exist on local machine
- download the Kaggle dataset automatically if they do not exist yet
- normalize the health labels into the binary classes `healthy` and `unhealthy`
- create stratified `train`, `val`, and `test` splits
- save a timestamped processed CSV into `data/processed/`

The generated file name looks like this:

```text
data/processed/processed_bee_data_YYYYMMDD_HHMMSS.csv
```

If you already downloaded the dataset yourself and only want to create the split file, run:

```powershell
py split.py --skip_dataset_download
```

If you want to force a fresh dataset copy from Kaggle, run:

```powershell
py split.py --force_dataset_download
```

### 4. Open the notebooks

After the split step, you can run the notebooks in `notebooks/`, `models/`, and `results/`.
It is the correct order to run the notebooks in `models/` step by step from the top.

Important: several notebooks currently reference a specific processed CSV file name with a fixed timestamp. If you generate a new processed file, update the CSV path inside those notebooks to match the newly created file in `data/processed/`.

In particular, check the notebooks in `models/` and `notebooks/exploratory.ipynb` before running them.

### Recommended run order

```text
1. py setup_env.py
2. .\.venv\Scripts\Activate.ps1
3. py split.py
4. Run the notebooks you want to inspect or train
```

## 3. Libraries Used
- pandas: loading CSV files, working with dataset tables. Used in split.py, cnn.ipynb, random_forest.ipynb, svm.ipynb, and exploratory.ipynb.
- numpy: array handling, label processing, numeric operations on image data and features. Used throughout the model notebooks
- scikit-learn: traditional ML models and preprocessing. It is used for train/validation/test splitting in split.py, label encoding in cnn.ipynb, and for Random Forest, SVM, scaling, grid search, and metrics in the model notebooks.
- tensorflow: CNN data pipeline, image loading, batching, preprocessing, and model training. Used in cnn.ipynb and cnn_svm_hybrid.ipynb.
keras: defining, training, saving, and reloading the CNN model. Used through TensorFlow Keras in cnn.ipynb and for loading the saved CNN in cnn_svm_hybrid.ipynb.
- opencv-python: image reading, resizing, color conversion, and histogram extraction for handcrafted features. Used in random_forest.ipynb and svm.ipynb.
- scikit-image: HOG feature extraction for the traditional ML models. Used in random_forest.ipynb and svm.ipynb.
- matplotlib: plotting images, training curves, and result charts. Used in all main notebooks, including cnn.ipynb, random_forest.ipynb, svm.ipynb, results.ipynb, and exploratory.ipynb.
- seaborn: cleaner statistical plots and visualizations. Used in cnn.ipynb, random_forest.ipynb, svm.ipynb, and cnn_svm_hybrid.ipynb.
- kagglehub: downloading the Kaggle bee dataset into the local project. Used in download_dataset.py.
These are even more secondary notebooks (along with Python standard library modules, but these are not listed here):

- Pillow: simple image loading for previewing sample bee images in the CNN notebook, used in cnn.ipynb.
- Jupyter and ipykernel: needed to run the notebooks, but not part of the bee-classification logic itself.

## 4. How to understand the project
This repository is the practical implementation of the methods described in the bachelor thesis. The thesis explains the researcher's motivation, Colony Collapse Disorder (CCD), the related work, the model choices, and the interpretation of the results. The repository shows how those ideas were implemented in code.

The project could be understood as a pipeline:

The environment is created with `setup_env.py`.
The dataset is downloaded and prepared with `split.py` and `download_dataset.py`.
The raw metadata in `data/raw/bee_data.csv` is converted into a processed split file in `data/processed`.
The notebooks in `notebooks` and `models` use that processed file for exploration, training, and evaluation.
The final plots and comparisons are summarized in `results/results.ipynb`.

### To understand reading order
To understand the repository step by step, it is advised to go through it in this order:

1. `split.py` to understand how the train, validation, and test split is created.
2. `notebooks/exploratory.ipynb` to understand the dataset and label distribution.
3. `models/cnn.ipynb` to understand the deep learning approach.
4. `models/random_forest.ipynb` and `models/svm.ipynb` to understand the traditional machine learning methods.
5. `models/cnn_svm_hybrid.ipynb` to understand the hybrid approach using CNN feature extraction with SVM classification.
6. `results/results.ipynb` to see the final comparison of results.

### Folder and file purpose
- `data`: stores the raw dataset, bee images, and processed metadata files.
- `notebooks`: exploratory analysis of the dataset.
- `models`: the implementation of the CNN, Random Forest, SVM, and hybrid CNN-SVM models.
- `results`: the final visual comparison and interpretation of model performance.
- `split.py`: creates the processed metadata file with the binary health labels and split groups.
- `download_dataset.py`: downloads and copies the Kaggle dataset into the local project structure (applied in `split.py`).
- `setup_env.py`: creates the virtual environment and installs the required dependencies.
- `requirements.txt`: contains used libraries required for the project to work

**Important implementation notice:**
The notebooks are connected. In particular, the processed CSV created by `split.py` is used as the source in the model notebooks, and the hybrid notebook depends on the CNN output model file saved by `models/cnn.ipynb`. Because of this, the notebooks should not be treated as completely independent files.

### Project Structure

```text
beeproject-cnn/
├── download_dataset.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup_env.py
├── split.py
├── data/
│   ├── bee_imgs/
│   ├── processed/
│   │   └── processed_bee_data_20260330_151740.csv
│   └── raw/
│       └── bee_data.csv
├── models/
│   ├── cnn_svm_hybrid.ipynb
│   ├── cnn.ipynb
│   ├── random_forest.ipynb
│   └── svm.ipynb
├── notebooks/
│   └── exploratory.ipynb
└── results/
    └── results.ipynb
```

## 5. Summary
In this project, four approaches are compared for classifying bee images into healthy and unhealthy classes: a Convolutional Neural Network (CNN), Random Forest, Support Vector Machine (SVM), and a hybrid CNN-SVM model. The goal was looking at their different performances, whether any traditional machine learning can compare in performance to CNN, and if a hybrid approach coulf be created to combine their strengths.

The core result is that the CNN achieved strongest overall performance, when taken the Recall and F1-Score metrics. The hybrid model also performs well and comes close to CNN, but it didn't outperform it in most cases (it sometimes did outperform CNN by +0.01 in Recall or F1-Score), and it seems like the performance of the Hybrid model heavily depends on the CNN feature extraction. This means that if the CNN model achieves a score, the Hybrid model will not produce a better result in most cases, but this could be examined in the future in more detail. 

The project also has a few limitations. The work is based on a single bee image dataset taken from Kaggle, so the results shouldn't be generalized broadly. The labels were also further simplified into binary healthy/unhealthy classes to further support the Research Questions in the thesis, and also because some health classes (e.g. Missing Queen) simply did not have enough samples to base projects upon. Reproducibility could be further improved, because it is simply too manual in some cases: downloading files from kaggle and CSV file references produced by `split.py`.

The hybrid idea was promising, but simply did not consistently surpass the standalone CNN model in performance. So the added complexity of combining the models is not yet totally justified, but could be further examined in future works. If the project were continued, it would be useful to improve automation, remove hardcoded file dependencies, test on additional datasets, and explore whether a stronger or more optimized hybrid setup can provide a clearer advantage. 

Something important mentioning, which is further explained in the thesis is that lots of similar images are present, which could have influenced the CNN model to rather focus on background colors and shapes than on the bees themselves. This is because cameras set up in front of hives will have the same setup and environment. For this reason, also further image segmentation should be explored. This is something that was not mentioned separately in related works.

As a final mention, AI models were used to explain specific concepts and code blocks found on the internet, and also to speed up code commenting. The outputs have all been supervised by the author (Me = Márió Palágyi) and only used after they were edited, if they proved to be useful. The models that were used are included in the GitHub Copilot Student Developer Pack: Claude Sonnet 4.6 for explaining concepts, GPT-5.3 Codex for understanding code blocks and commenting.