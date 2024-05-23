import threading
import multiprocessing
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import os

# Constants
DATASET_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv",
]
DATASET_NAMES = ["diabetes.csv", "wine.csv"]
DOWNLOAD_DIR = "datasets"
NUM_THREADS = 4
NUM_PROCESSES = 4

# Function to download a dataset
def download_dataset(url, dataset_name, results, index):
    try:
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        with open(f"{DOWNLOAD_DIR}/{dataset_name}", "wb") as file:
            file.write(response.content)
        results[index] = f"{DOWNLOAD_DIR}/{dataset_name}"
        print(f"Downloaded {dataset_name}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")

# Function to preprocess a dataset
def preprocess_dataset(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        if "diabetes" in dataset_path:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        elif "wine" in dataset_path:
            df.columns = [
                "Class",
                "Alcohol",
                "Malic_Acid",
                "Ash",
                "Alcalinity_of_Ash",
                "Magnesium",
                "Total_Phenols",
                "Flavanoids",
                "Nonflavanoid_Phenols",
                "Proanthocyanins",
                "Color_Intensity",
                "Hue",
                "OD280/OD315_of_Diluted_Wines",
                "Proline"
            ]
            X = df.iloc[:, 1:]  # Features
            y = df.iloc[:, 0]   # Target (Class)
        print(f"Preprocessed {dataset_path}")
        return (X, y)
    except Exception as e:
        print(f"Failed to preprocess {dataset_path}: {e}")
        return None

# Function to train a machine learning model
def train_model(data, index):
    print(f"Starting training for dataset index {index}")
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained for dataset index {index} with accuracy: {accuracy:.2f}%")
    return accuracy

# Main function to orchestrate downloading, preprocessing, and training
def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Download datasets using threads
    download_threads = []
    download_results = [None] * len(DATASET_URLS)
    for i in range(len(DATASET_URLS)):
        thread = threading.Thread(target=download_dataset, args=(DATASET_URLS[i], DATASET_NAMES[i], download_results, i))
        download_threads.append(thread)
        thread.start()

    for thread in download_threads:
        thread.join()

    print("All datasets downloaded.")

    # Preprocess datasets using multiprocessing
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        preprocess_results = pool.map(preprocess_dataset, download_results)

    print("All datasets preprocessed.")
    print("Preprocess results:", preprocess_results)

    # Filter out failed preprocessing results
    valid_results = [(i, data) for i, data in enumerate(preprocess_results) if data is not None]

    # Train models using parallel processing
    accuracies = Parallel(n_jobs=NUM_PROCESSES)(delayed(train_model)(data, i) for i, data in valid_results)

    # Print accuracies
    for (i, _), accuracy in zip(valid_results, accuracies):
        print(f"Model accuracy for dataset {DATASET_NAMES[i]}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
