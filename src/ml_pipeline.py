import argparse
import io
import pickle
import uuid
from typing import List

import fsspec
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def ensure_dir_exists(fs, path):
    """Ensure directory exists using fsspec filesystem."""
    try:
        fs.makedirs(path, exist_ok=True)
    except Exception:
        # Some filesystems might not support makedirs
        pass


def join_path(base_path, *paths):
    """Join paths in a filesystem-agnostic way."""
    if base_path.endswith("/"):
        base_path = base_path.rstrip("/")

    result = base_path
    for path in paths:
        if isinstance(path, str):
            path = path.lstrip("/")
            result = f"{result}/{path}"
        else:
            result = f"{result}/{str(path)}"
    return result


class ScalerParameters(BaseModel):
    feature_range: List[float]


class ScalerConfig(BaseModel):
    type: str
    parameters: ScalerParameters


class ModelParameters(BaseModel):
    max_iter: int
    C: float
    solver: str
    penalty: str


class ModelConfig(BaseModel):
    parameters: ModelParameters


class PipelineParams(BaseSettings):
    random_state: int
    test_size: float
    id_column: str
    target_column: str
    feature_names: List[str]
    scaler: ScalerConfig
    model: ModelConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            JsonConfigSettingsSource(settings_cls),
        )

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",  # prefix for env vars
        env_nested_delimiter="__",
    )


def read_data(file_path):
    """Read acidity data from CSV file."""
    with fsspec.open(file_path, "r") as f:
        return pd.read_csv(f, sep=";")


def parse_id_column(df, id_column):
    """Parse ID column by removing dashes."""
    df_copy = df.copy()
    df_copy.loc[:, id_column] = df_copy.loc[:, id_column].map(
        lambda x: x.replace("-", "")
    )
    return df_copy


def merge_data(acidity_df, other_df, id_column):
    """Merge acidity and other dataframes on id column and drop id."""
    merged_data = acidity_df.merge(other_df, on=id_column)
    return merged_data.drop(id_column, axis="columns")


def calculate_correlation(data):
    """Calculate correlation matrix for the dataset."""
    return data.corr()


def describe_dataset(data):
    """Generate descriptive statistics for the dataset."""
    return data.describe()


def split_dataset(X, y, test_size, random_state):
    """Split dataset into train and test sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def generate_correlation_plot(corr_matrix, output_dir):
    """Generate and save correlation plot."""
    fig, ax = plt.subplots(figsize=(10, 10))

    cax = ax.matshow(corr_matrix, cmap="viridis")
    fig.colorbar(cax)

    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(
                j,
                i,
                round(corr_matrix.iloc[i, j], 1),
                ha="center",
                va="center",
                color="w",
            )

    plt.tight_layout()
    output_path = join_path(output_dir, "correlation-plot.jpeg")

    # Save plot using fsspec
    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg", dpi=150)
    buffer.seek(0)

    with fsspec.open(output_path, "wb") as f:
        f.write(buffer.getvalue())

    plt.close()
    buffer.close()
    return output_path


def select_features(X, feature_names):
    """Select specific features from the dataset."""
    return X[feature_names]


def fit_scaler(X_train, scaler_config: ScalerConfig):
    """Fit scaler on training data."""
    if scaler_config.type == "MinMaxScaler":
        scaler = MinMaxScaler(
            feature_range=tuple(scaler_config.parameters.feature_range)
        )
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_config.type}")

    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled


def scale_data(scaler, X):
    """Scale data using fitted scaler."""
    return scaler.transform(X)


def train_model(X_train_scaled, y_train, random_state, model_parameters):
    """Train logistic regression model."""
    model = LogisticRegression(random_state=random_state, **model_parameters)
    model.fit(X_train_scaled, y_train)
    return model


def predict_using_model(model, X):
    """Make predictions using trained model."""
    return model.predict(X)


def calculate_metrics(y_true, y_pred):
    """Calculate and return classification metrics."""
    return classification_report(y_true, y_pred)


# Serialized wrapper functions
def load_and_preprocess_data_serialized(
    data_dir: str, acidity_path: str, other_path: str, params: PipelineParams
):
    """Wrapper for initial data loading and preprocessing that handles serialization."""
    # Execute original functions
    data_acidity = read_data(acidity_path)
    data_other = read_data(other_path)
    data_other_parsed = parse_id_column(data_other, params.id_column)

    # Serialize initial data for merge step
    acidity_path = join_path(data_dir, "acidity_data.csv")
    other_path = join_path(data_dir, "other_data_parsed.csv")

    with fsspec.open(acidity_path, "w") as f:
        data_acidity.to_csv(f, index=False, sep=";")

    with fsspec.open(other_path, "w") as f:
        data_other_parsed.to_csv(f, index=False, sep=";")


def explore_data_serialized(data_dir: str, output_dir: str):
    """Wrapper for data exploration that handles serialization."""
    # Read data from data_dir
    merged_data_path = join_path(data_dir, "merged_data.csv")
    with fsspec.open(merged_data_path, "r") as f:
        merged_data = pd.read_csv(f)

    # Execute original functions
    description = describe_dataset(merged_data)
    corr_matrix = calculate_correlation(merged_data)
    plot_path = generate_correlation_plot(corr_matrix, output_dir)

    # Serialize results
    description_path = join_path(data_dir, "dataset_description.txt")
    with fsspec.open(description_path, "w") as f:
        f.write(str(description))

    return description, plot_path


def merge_data_serialized(data_dir: str, params: PipelineParams):
    """Wrapper for merge_data that handles serialization."""
    # Read data from data_dir
    acidity_path = join_path(data_dir, "acidity_data.csv")
    other_path = join_path(data_dir, "other_data_parsed.csv")

    with fsspec.open(acidity_path, "r") as f:
        acidity_df = pd.read_csv(f, sep=";")

    with fsspec.open(other_path, "r") as f:
        other_df = pd.read_csv(f, sep=";")

    # Execute original function
    merged_data = merge_data(acidity_df, other_df, params.id_column)

    # Serialize result
    merged_path = join_path(data_dir, "merged_data.csv")
    with fsspec.open(merged_path, "w") as f:
        merged_data.to_csv(f, index=False)


def split_dataset_serialized(data_dir: str, params: PipelineParams):
    """Wrapper for split_dataset that handles serialization."""
    # Read data from data_dir
    merged_path = join_path(data_dir, "merged_data.csv")
    with fsspec.open(merged_path, "r") as f:
        merged_data = pd.read_csv(f)

    X = merged_data.drop(params.target_column, axis="columns")
    y = merged_data[params.target_column]

    # Execute original function
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, params.test_size, params.random_state
    )

    # Serialize results
    X_train_path = join_path(data_dir, "X_train.csv")
    X_test_path = join_path(data_dir, "X_test.csv")
    y_train_path = join_path(data_dir, "y_train.csv")
    y_test_path = join_path(data_dir, "y_test.csv")

    with fsspec.open(X_train_path, "w") as f:
        X_train.to_csv(f, index=False)
    with fsspec.open(X_test_path, "w") as f:
        X_test.to_csv(f, index=False)
    with fsspec.open(y_train_path, "w") as f:
        y_train.to_csv(f, index=False)
    with fsspec.open(y_test_path, "w") as f:
        y_test.to_csv(f, index=False)


def select_features_serialized(
    data_dir: str, input_filename: str, output_filename: str, params: PipelineParams
):
    """Wrapper for select_features that handles serialization with configurable file names."""
    # Read data from data_dir
    input_path = join_path(data_dir, input_filename)

    with fsspec.open(input_path, "r") as f:
        X_data = pd.read_csv(f)

    # Execute original function
    X_selected = select_features(X_data, params.feature_names)

    # Serialize result
    output_path = join_path(data_dir, output_filename)

    with fsspec.open(output_path, "w") as f:
        X_selected.to_csv(f, index=False)


def fit_scaler_serialized(data_dir: str, scaler_config: ScalerConfig):
    """Wrapper for fit_scaler that handles serialization - only fits and saves scaler."""
    # Read data from data_dir
    X_train_selected_path = join_path(data_dir, "X_train_selected.csv")
    with fsspec.open(X_train_selected_path, "r") as f:
        X_train_selected = pd.read_csv(f)

    # Execute original function - only fit the scaler
    if scaler_config.type == "MinMaxScaler":
        scaler = MinMaxScaler(
            feature_range=tuple(scaler_config.parameters.feature_range)
        )
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_config.type}")

    scaler.fit(X_train_selected)

    # Serialize scaler only
    scaler_path = join_path(data_dir, "scaler.pkl")
    with fsspec.open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)


def scale_data_serialized(data_dir: str, input_filename: str, output_filename: str):
    """Wrapper for scale_data that handles serialization with configurable file names."""
    # Read data from data_dir
    input_path = join_path(data_dir, input_filename)
    scaler_path = join_path(data_dir, "scaler.pkl")

    with fsspec.open(input_path, "r") as f:
        X_data = pd.read_csv(f)
    with fsspec.open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Execute original function
    X_scaled = scale_data(scaler, X_data)

    # Serialize result
    output_path = join_path(data_dir, output_filename)
    with fsspec.open(output_path, "w") as f:
        pd.DataFrame(X_scaled, columns=X_data.columns).to_csv(f, index=False)
    """Wrapper for scale_data that handles serialization."""
    # Read data from data_dir
    X_test_selected_path = join_path(data_dir, "X_test_selected.csv")
    scaler_path = join_path(data_dir, "scaler.pkl")

    with fsspec.open(X_test_selected_path, "r") as f:
        X_test_selected = pd.read_csv(f)
    with fsspec.open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Execute original function
    X_test_scaled = scale_data(scaler, X_test_selected)

    # Serialize result
    X_test_scaled_path = join_path(data_dir, "X_test_scaled.csv")
    with fsspec.open(X_test_scaled_path, "w") as f:
        pd.DataFrame(X_test_scaled, columns=X_test_selected.columns).to_csv(
            f, index=False
        )


def train_model_serialized(data_dir: str, params: PipelineParams):
    """Wrapper for train_model that handles serialization."""
    # Read data from data_dir
    X_train_scaled_path = join_path(data_dir, "X_train_scaled.csv")
    y_train_path = join_path(data_dir, "y_train.csv")

    with fsspec.open(X_train_scaled_path, "r") as f:
        X_train_scaled = pd.read_csv(f)
    with fsspec.open(y_train_path, "r") as f:
        y_train = pd.read_csv(f).values.ravel()

    # Execute original function
    model = train_model(
        X_train_scaled,
        y_train,
        params.random_state,
        params.model.parameters.model_dump(),
    )

    # Serialize result
    model_path = join_path(data_dir, "model.pkl")
    with fsspec.open(model_path, "wb") as f:
        pickle.dump(model, f)


def predict_using_model_serialized(
    data_dir: str, input_filename: str, output_filename: str
):
    """Wrapper for predict_using_model that handles serialization with configurable file names."""
    # Read data from data_dir
    input_path = join_path(data_dir, input_filename)
    model_path = join_path(data_dir, "model.pkl")

    with fsspec.open(input_path, "r") as f:
        X_data = pd.read_csv(f)
    with fsspec.open(model_path, "rb") as f:
        model = pickle.load(f)

    # Execute original function
    y_pred = predict_using_model(model, X_data)

    # Serialize result
    output_path = join_path(data_dir, output_filename)

    with fsspec.open(output_path, "w") as f:
        pd.DataFrame(y_pred, columns=["prediction"]).to_csv(f, index=False)


def calculate_metrics_serialized(
    data_dir: str, y_true_filename: str, y_pred_filename: str, output_filename: str
):
    """Wrapper for calculate_metrics that handles serialization with configurable file names."""
    # Read data from data_dir
    y_true_path = join_path(data_dir, y_true_filename)
    y_pred_path = join_path(data_dir, y_pred_filename)

    with fsspec.open(y_true_path, "r") as f:
        y_true = pd.read_csv(f).values.ravel()
    with fsspec.open(y_pred_path, "r") as f:
        y_pred = pd.read_csv(f).values.ravel()

    # Execute original function
    metrics = calculate_metrics(y_true, y_pred)

    # Serialize result
    output_path = join_path(data_dir, output_filename)

    with fsspec.open(output_path, "w") as f:
        f.write(metrics)

    return metrics


def generate_run_id():
    """Generate a unique run ID for this pipeline execution."""
    return uuid.uuid4().hex


def main():
    """Main pipeline function orchestrating all steps."""
    tasks_to_execute = [
        "load_data",
        "merge_data",
        "explore_data",
        "split_dataset",
        "select_features_train",
        "select_features_test",
        "fit_scaler",
        "scale_train",
        "scale_test",
        "train_model",
        "predict_train",
        "predict_test",
        "evaluate_train",
        "evaluate_test",
    ]

    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="Wine Quality ML Pipeline")
    parser.add_argument(
        "--acidity-path",
        type=str,
        required=True,
        help="Path to the wine acidity CSV file",
    )
    parser.add_argument(
        "--other-path",
        type=str,
        required=True,
        help="Path to the wine other attributes CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save outputs (plots, data, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=f"Comma-separated list of tasks to execute. Available tasks: {', '.join(tasks_to_execute)}. If not specified, all tasks will be executed.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID for this pipeline execution. If not provided, a unique ID will be generated.",
    )
    args = parser.parse_args()

    # Parse tasks to execute
    if args.task is not None:
        tasks_to_execute = [task.strip() for task in args.task.split(",")]

    # Create output directory if it doesn't exist using fsspec
    output_dir = args.output_dir
    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = generate_run_id()
    output_dir = output_dir.rstrip("/") + "/" + run_id

    fs = fsspec.url_to_fs(output_dir)
    ensure_dir_exists(fs, output_dir)

    # Load configuration from JSON file
    PipelineParams.model_config["json_file"] = args.config
    params = PipelineParams()
    print(params.model_dump_json(indent=4))

    # Data loading and initial preprocessing (serialized)
    if "load_data" in tasks_to_execute:
        print("Loading data...")
        load_and_preprocess_data_serialized(
            output_dir, args.acidity_path, args.other_path, params
        )

    # Merge data (serialized)
    if "merge_data" in tasks_to_execute:
        print("Merging data...")
        merge_data_serialized(output_dir, params)

    # Data exploration (serialized)
    if "explore_data" in tasks_to_execute:
        print("Exploring data...")
        description, plot_path = explore_data_serialized(output_dir, output_dir)
        print("\nDataset description:")
        print(description)
        print(f"\nCorrelation plot saved as {plot_path}")

    # Split dataset (serialized)
    if "split_dataset" in tasks_to_execute:
        print("\nSplitting dataset...")
        split_dataset_serialized(output_dir, params)

    # Feature selection (serialized)
    if "select_features_train" in tasks_to_execute:
        print("Selecting features for training data...")
        select_features_serialized(
            output_dir, "X_train.csv", "X_train_selected.csv", params
        )

    if "select_features_test" in tasks_to_execute:
        print("Selecting features for test data...")
        select_features_serialized(
            output_dir, "X_test.csv", "X_test_selected.csv", params
        )

    # Feature scaling (serialized)
    if "fit_scaler" in tasks_to_execute:
        print("Fitting scaler...")
        fit_scaler_serialized(output_dir, params.scaler)

    if "scale_train" in tasks_to_execute:
        print("Scaling training data...")
        scale_data_serialized(output_dir, "X_train_selected.csv", "X_train_scaled.csv")

    if "scale_test" in tasks_to_execute:
        print("Scaling test data...")
        scale_data_serialized(output_dir, "X_test_selected.csv", "X_test_scaled.csv")

    # Model training (serialized)
    if "train_model" in tasks_to_execute:
        print("Training model...")
        train_model_serialized(output_dir, params)

    # Model evaluation (serialized)
    if "predict_train" in tasks_to_execute:
        print("Making predictions on training data...")
        predict_using_model_serialized(
            output_dir, "X_train_scaled.csv", "y_pred_train.csv"
        )

    if "predict_test" in tasks_to_execute:
        print("Making predictions on test data...")
        predict_using_model_serialized(
            output_dir, "X_test_scaled.csv", "y_pred_test.csv"
        )

    if "evaluate_train" in tasks_to_execute:
        print("Evaluating model on training data...")
        train_metrics = calculate_metrics_serialized(
            output_dir, "y_train.csv", "y_pred_train.csv", "train_metrics.txt"
        )
        print("\nTrain metrics:")
        print(train_metrics)

    if "evaluate_test" in tasks_to_execute:
        print("Evaluating model on test data...")
        test_metrics = calculate_metrics_serialized(
            output_dir, "y_test.csv", "y_pred_test.csv", "test_metrics.txt"
        )
        print("\nTest metrics:")
        print(test_metrics)


if __name__ == "__main__":
    main()
