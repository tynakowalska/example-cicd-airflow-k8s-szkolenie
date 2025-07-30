import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml_pipeline import ScalerConfig, ScalerParameters, fit_scaler


def test_can_fit_scaler():
    # Arrange
    X_train = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    scaler_config = ScalerConfig(
        type="MinMaxScaler", parameters=ScalerParameters(feature_range=[0.0, 1.0])
    )

    # Act
    scaler, X_train_scaled = fit_scaler(X_train, scaler_config)

    # Assert - Check scaler class
    assert isinstance(scaler, MinMaxScaler)
    assert scaler.feature_range == (0.0, 1.0)

    # Assert - Check scaled data properties
    assert X_train_scaled.shape == X_train.shape
    assert np.all(X_train_scaled >= 0.0)
    assert np.all(X_train_scaled <= 1.0)

    # Assert - Check that min/max values are correctly scaled
    assert np.allclose(X_train_scaled.min(axis=0), 0.0)
    assert np.allclose(X_train_scaled.max(axis=0), 1.0)
