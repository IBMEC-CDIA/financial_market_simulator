# LSTM autoencoder

Autoencoder recorrente (LSTM) que detecta janelas incomuns nos log-retornos
diários de um ativo. O erro de reconstrução de cada janela é o escore de
anomalia: uma janela que o modelo reconstrói mal é marcada como anômala.

O pacote segue o padrão de projeto de
[`specs/project_structure_check/`](../../../../specs/project_structure_check/README.md)
e não importa nada de outro pacote de modelo (spec 01, CA-04).

## Arquitetura

O `LSTMAutoencoder` (`autoencoder.py`) recebe e devolve tensores de shape
`(batch, window_size)`, o mesmo formato do `SlidingWindowReconstructionDataset`.

| Bloco | Camadas, na ordem |
|---|---|
| Codificador | `LSTM(1, hidden_layer_size)`, lendo um valor por passo de tempo; `Linear(hidden_layer_size, latent_layer_size)` sobre o último estado oculto |
| Decodificador | vetor latente repetido em `window_size` passos; `LSTM(latent_layer_size, hidden_layer_size)`; `Linear(hidden_layer_size, 1)` em cada passo |

Não há ativação depois das camadas `Linear`, por isso o vetor latente e a
reconstrução podem ser negativos, como os log-retornos normalizados.

## Módulos

| Módulo | Conteúdo | Spec |
|---|---|---|
| `autoencoder.py` | `LSTMAutoencoder` | - |
| `config.py` | `LSTMAutoencoderTrainingConfig` e as sub-configurações | - |
| `trainer.py` | `LSTMAutoencoderTrainer`, `LSTMAutoencoderTrainingResult` e `compute_reconstruction_scores` | 03, 06 |
| `trainer_factory.py` | `build_trainer(ticker, price_data, options)` | 06 |
| `overfitting.py` | relatório `*-overfitting.json` e comando de avaliação | 04 |
| `run_train_script.py` | treino de um ticker pela linha de comando | 02 |

Os testes ficam em `tests/generative_models/models/lstm_autoencoder/`, com um
`test_<módulo>.py` para cada módulo.

## Hiperparâmetros padrão

| Parâmetro | Valor padrão |
|---|---|
| Janela (`--window-size`) | 10 log-retornos |
| Estado oculto das LSTM / vetor latente | 16 / 4 |
| Lote (`--batch-size`) | 32 |
| Épocas (`--epochs`) | 300 |
| Taxa de aprendizado (`--learning-rate`) | 0,001 (Adam, perda MSE) |
| Semente (`--random-seed`) | 42 |
| Contaminação (`--contamination`) | 0,05 |
| Divisão treino / validação / teste | 70% / 15% / 15% |
| Projeto no wandb (`--project-name`) | `autoencoder-lstm-outlier-trading-agent` |

## Dados

O treinador recebe os preços no formato da spec 03 (`close_price`,
`log_return` e `split`). Ele usa a coluna `split` como está e não recalcula
a divisão. Um DataFrame sem `close_price` ou sem `split` gera `ValueError`.
A normalização usa apenas a média e o desvio padrão do conjunto de treino, e
o conjunto de teste é avaliado uma única vez, depois do treino.

Sem `price_data`, o treinador baixa os preços com `fetch_price_series` e
aplica as proporções de `MarketDataConfig`.

## Como treinar

Vários tickers, pelo comando compartilhado (o mesmo que o `train.yml` chama
com `model=lstm_autoencoder`):

```bash
uv run python -m generative_models.models.run_model_training \
    --model lstm_autoencoder --tickers AAPL,MSFT --epochs 300 \
    --learning-rate 0.001 --window-size 10 --device cpu \
    --output-directory artifacts
```

Um ticker, a partir do CSV gravado por `collect_market_data`:

```bash
uv run python -m generative_models.data.collect_market_data \
    --ticker AAPL --output-path artifacts/market-data/AAPL.csv

uv run python -m generative_models.models.lstm_autoencoder.run_train_script \
    --ticker AAPL --price-data-path artifacts/market-data/AAPL.csv \
    --epochs 300 --device cpu --output-directory artifacts
```

Sem `--price-data-path`, o comando baixa os preços entre `--start-date` e
`--end-date`. Para ver todas as opções, use `--help`.

Com `WANDB_MODE=disabled`, o treino roda sem chave e sem enviar nada ao
wandb. Com a chave `WANDB_API_KEY` no arquivo `.env`, o run e o modelo são
registrados no projeto do wandb, e o modelo recebe o alias de
`--experiment-name`.

## Arquivos gravados

Na pasta `--output-directory`, para cada ticker:

- `lstm-autoencoder-<ticker>.pt`: o `state_dict` do `LSTMAutoencoder`;
- `lstm-autoencoder-<ticker>-overfitting.json`: o relatório da spec 04.

O nome do artifact no wandb é o mesmo do arquivo, sem a extensão:
`lstm-autoencoder-<ticker>`.

## Como avaliar

```bash
uv run python -m generative_models.models.lstm_autoencoder.overfitting \
    --report-path artifacts/lstm-autoencoder-aapl-overfitting.json
```

O comando sai com código `0` quando `overfitting_ratio` é menor ou igual a
`1.5` (ou ao valor de `--max-overfitting-ratio`) e `test_loss` é menor que
`baseline_test_loss`. Caso contrário, sai com código `1`.

## Notebook

`notebooks/autoencoder_outlier.ipynb` usa este pacote para treinar o modelo de
um ticker, gravar o relatório de overfitting e gerar os gráficos de outliers e
de sinais de compra e venda.

## Limites conhecidos

- O serving ainda não carrega este modelo. O registro local e o do wandb
  reconstroem apenas o `SimpleAutoencoder` do pacote `base_autoencoder`
  (spec 08, seção Limites). Um arquivo `lstm-autoencoder-*.pt` colocado na
  pasta `models/` não é servido: a API falha ao tentar carregá-lo como
  `SimpleAutoencoder`. Para servir o modelo LSTM, é preciso um leitor próprio
  em `serving/`, em um PR separado (spec 07).
- No relatório de overfitting, `hidden_layer_size` é o tamanho do estado
  oculto das camadas LSTM, e não o de uma camada densa.
- O modelo usa uma única camada LSTM no codificador e outra no decodificador.
  O número de camadas não é configurável.
- A saída é um escore de anomalia, e não uma previsão de preço. Os resultados
  dependem do período e do ticker, e não foram validados para negociação.
