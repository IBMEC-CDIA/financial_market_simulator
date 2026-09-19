"""Train one dense autoencoder per ticker symbol.

Example
-------
    uv run --no-sync python scripts/train_autoencoders.py
"""

from generative_models.models.autoencoder.autoencoder_ticker_model_manager import (  # pylint: disable=line-too-long
    AutoencoderTickerModelManager,
    AutoencoderTrainingConfig,
)

TICKERS = {
    "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
    "MSFT": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
    "NVDA": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
    "GOOGL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
}

TRAINING_CONFIG = AutoencoderTrainingConfig(
    window_size=10,
    batch_size=32,
    number_of_epochs=100,
    learning_rate=1e-3,
    hidden_layer_size=16,
    latent_layer_size=4,
    validation_fraction=0.2,
    random_seed=42,
)

PROJECT_NAME = "autoencoder-ticker-models"


def main() -> AutoencoderTickerModelManager:
    """Train the autoencoder of every ticker in `TICKERS`.

    Returns
    -------
    AutoencoderTickerModelManager
        Manager holding the trained models and training results.
    """
    manager = AutoencoderTickerModelManager(
        tickers=TICKERS,
        training_config=TRAINING_CONFIG,
        project_name=PROJECT_NAME,
    )

    manager.train_all_models()

    for ticker, training_result in manager.training_results.items():
        validation_loss = training_result.validation_loss
        print(f"{ticker} validation MSE: {validation_loss:.6f}")

    return manager


if __name__ == "__main__":
    main()
