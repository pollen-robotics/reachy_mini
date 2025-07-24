import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew
from tqdm import tqdm

tqdm.pandas()

import argparse
from pathlib import Path

import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for training and testing with data folders."
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default="touch_classifier.joblib",
        help="Path to save the trained model (default: touch_classifier.joblib)",
    )

    parser.add_argument(
        "--train-data-folder",
        type=Path,
        required=True,
        help="Path to the training data folder",
    )

    parser.add_argument(
        "--test-data-folder",
        type=Path,
        required=True,
        help="Path to the testing data folder",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for audio files (default: 16000)",
    )

    return parser.parse_args()


def main():
    """
    Touch classification script.
    Loads audio files from specified folders, extracts features, trains a model, and evaluates it on test data.
    """

    args = parse_args()

    # Load data
    audio_train_files = list(args.train_data_folder.glob("*.wav"))
    audio_test_files = list(args.test_data_folder.glob("*.wav"))

    print(f"Number of training files: {len(audio_train_files)}")
    print(f"Number of testing files: {len(audio_test_files)}")

    # Generate mfcc features
    def get_mfcc(file_name, sample_rate, mfcc_number=30):
        data, _ = librosa.core.load(file_name, sr=sample_rate)

        ft1 = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=mfcc_number)
        ft2 = librosa.feature.zero_crossing_rate(y=data)[0]
        ft3 = librosa.feature.spectral_rolloff(y=data)[0]
        ft4 = librosa.feature.spectral_centroid(y=data)[0]
        ft5 = librosa.feature.spectral_contrast(y=data)[0]
        ft6 = librosa.feature.spectral_bandwidth(y=data)[0]

        ft1_trunc = np.hstack(
            (
                np.mean(ft1, axis=1),
                np.std(ft1, axis=1),
                skew(ft1, axis=1),
                np.max(ft1, axis=1),
                np.median(ft1, axis=1),
                np.min(ft1, axis=1),
            )
        )
        ft2_trunc = np.hstack(
            (
                np.mean(ft2),
                np.std(ft2),
                skew(ft2),
                np.max(ft2),
                np.median(ft2),
                np.min(ft2),
            )
        )
        ft3_trunc = np.hstack(
            (
                np.mean(ft3),
                np.std(ft3),
                skew(ft3),
                np.max(ft3),
                np.median(ft3),
                np.min(ft3),
            )
        )
        ft4_trunc = np.hstack(
            (
                np.mean(ft4),
                np.std(ft4),
                skew(ft4),
                np.max(ft4),
                np.median(ft4),
                np.min(ft4),
            )
        )
        ft5_trunc = np.hstack(
            (
                np.mean(ft5),
                np.std(ft5),
                skew(ft5),
                np.max(ft5),
                np.median(ft5),
                np.min(ft5),
            )
        )
        ft6_trunc = np.hstack(
            (
                np.mean(ft6),
                np.std(ft6),
                skew(ft6),
                np.max(ft6),
                np.median(ft6),
                np.max(ft6),
            )
        )

        return pd.Series(
            np.hstack(
                (ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)
            )
        )

    # Prepare data
    train_data = pd.DataFrame()
    train_data["file_name"] = audio_train_files
    test_data = pd.DataFrame()
    test_data["file_name"] = audio_test_files

    train_data = train_data["file_name"].progress_apply(
        get_mfcc, sample_rate=args.sample_rate
    )
    test_data = test_data["file_name"].progress_apply(
        get_mfcc, sample_rate=args.sample_rate
    )

    train_data["file_name"] = audio_train_files
    test_data["file_name"] = audio_test_files

    train_data["label"] = train_data["file_name"].progress_apply(
        lambda f: f.stem.split("_")[0]
    )
    test_data["label"] = test_data["file_name"].progress_apply(
        lambda f: f.stem.split("_")[0]
    )

    # Construct features set
    # We want to create a classifier that predicts the index train_index based on the features x
    train_features = train_data.drop(["label", "file_name"], axis=1)
    train_features = train_features.values
    labels = np.sort(np.unique(train_data.label.values))

    label_to_index = {}
    index_to_label = {}
    for index, label in enumerate(labels):
        label_to_index[label] = index
        index_to_label[index] = label
    train_index = np.array([label_to_index[x] for x in train_data.label.values])

    test_features = test_data.drop(["label", "file_name"], axis=1)
    test_features = test_features.values

    # Apply scaling for PCA
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Apply PCA for dimension reduction
    # pca = PCA(n_components=22).fit(train_features_scaled)
    pca = PCA(n_components=0.95).fit(train_features_scaled)
    train_features_pca = pca.transform(train_features_scaled)
    test_features_pca = pca.transform(test_features_scaled)

    print("Explained variance ratio :", sum(pca.explained_variance_ratio_))

    # Fit an SVM model
    train_features_split, test_features_split, train_index_split, test_index_split = (
        train_test_split(
            train_features_pca,
            train_index,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
    )

    clf = SVC(kernel="rbf", probability=True)

    clf.fit(train_features_split, train_index_split)

    print(
        "Accuracy score :",
        accuracy_score(clf.predict(test_features_split), test_index_split),
    )

    # Define the paramter grid for C from 0.001 to 10, gamma from 0.001 to 10
    C_grid = [0.001, 0.01, 0.1, 1, 10]
    gamma_grid = [0.001, 0.01, 0.1, 1, 10]
    param_grid = {"C": C_grid, "gamma": gamma_grid}

    grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=3, scoring="accuracy")
    grid.fit(train_features_split, train_index_split)

    # Optimal model
    clf = SVC(
        kernel="rbf",
        C=grid.best_params_["C"],
        gamma=grid.best_params_["gamma"],
        probability=True,
    )
    clf.fit(train_features_split, train_index_split)

    # Save the model, scaler, pca, and label mappings
    model_bundle = {
        "model": clf,
        "scaler": scaler,
        "pca": pca,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
    }
    joblib.dump(model_bundle, args.model_output)
    print(f"Model saved to {args.model_output}")

    print(
        "Accuracy score :",
        accuracy_score(clf.predict(test_features_split), test_index_split),
    )

    predictions = clf.predict_proba(test_features_pca)

    for prediction, label in zip(predictions, test_data["label"]):
        idx = np.argsort(prediction)[::-1]
        print(f"Top 3 predictions: {[index_to_label[i] for i in idx[:3]]}")
        print(sorted(prediction)[::-1])
        print(f"Actual label: {label}")


if __name__ == "__main__":
    main()
