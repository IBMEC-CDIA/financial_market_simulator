# 04 - Script de predição

- **Arquivo:** `src/generative_models/serving/predict_autoencoder.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_predict_autoencoder.py`
- **Depende de:** specs 01 e 03, `training/01`
- **Situação:** a implementar

## Objetivo

Validar pelo terminal um modelo registrado no wandb: carregar o
modelo mais recente (ou o de um ticker), baixar os preços recentes
desse ticker, pontuar todas as janelas e imprimir um resumo.

```bash
uv run python -m generative_models.serving.predict_autoencoder
uv run python -m generative_models.serving.predict_autoencoder \
    --ticker PETR4.SA --start-date 2026-01-01 --top-outliers 10
```

As funções `load_predictor` e `predict_period` também são reutilizadas
pelo serviço de modelos (spec 05) e pela API (spec 07).

## Origem

- Notebook, células 32 a 35: pontuação da série, contagem de
  outliers e alinhamento de cada janela com sua data.
- `training/08`: mesmo padrão de script com `build_argument_parser` e
  `main(command_line_arguments)`.

## Interface pública

```python
def build_argument_parser() -> argparse.ArgumentParser:
    ...


def build_prediction_table(
    price_data: pd.DataFrame,
    prediction: AutoencoderPrediction,
) -> pd.DataFrame:
    ...


def load_predictor(
    registry: WandbAutoencoderRegistry,
    ticker: str | None = None,
) -> AutoencoderPredictor:
    ...


def predict_period(
    predictor: AutoencoderPredictor,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    ...


def main(
    command_line_arguments: list[str] | None = None,
) -> pd.DataFrame:
    ...
```

O módulo termina com `if __name__ == "__main__": main()`.
`fetch_price_series` e `WandbAutoencoderRegistry` devem ser importados
no topo do módulo, para que os testes possam substituí-los com
`patch("generative_models.serving.predict_autoencoder.<nome>")`.

## Comportamento

### `build_argument_parser`

| Argumento | Tipo | Padrão |
|---|---|---|
| `--ticker` | `str` | `None` (modelo mais recente) |
| `--project-name` | `str` | `TrackingConfig.project_name` |
| `--entity` | `str` | `None` |
| `--start-date` | `str` | hoje menos 365 dias (`YYYY-MM-DD`) |
| `--end-date` | `str` | hoje (`YYYY-MM-DD`) |
| `--top-outliers` | `int` | `5` |
| `--device` | `str` | `None` |

### `build_prediction_table`

- Recebe os preços (`DataFrame` com `close_price`, indexado por data,
  como em `training/01`) e a predição feita **a partir desses
  preços**.
- O índice final `i` de uma janela se refere ao retorno entre os
  preços `i` e `i + 1`. Por isso, a linha de preço da janela é
  `i + 1`.
- Retorna um `DataFrame` com uma linha por janela, indexado pela data
  da linha de preço `i + 1`, com as colunas `close_price`,
  `reconstruction_score` e `is_outlier`, nesta ordem.

### `load_predictor`

- Com `ticker` (não vazio), retorna `registry.load_model(ticker)`.
- Sem `ticker`, retorna `registry.load_latest_model()`.

### `predict_period`

- Busca os preços de `predictor.metadata.ticker` com
  `fetch_price_series(ticker, start_date, end_date)`, chama
  `predictor.predict_from_prices` com `close_price` e retorna
  `build_prediction_table`.

### `main`

1. Faz o parse dos argumentos.
2. Cria `WandbAutoencoderRegistry(project_name=..., entity=...,
   device=...)`.
3. Carrega o preditor com `load_predictor(registry, ticker)`.
4. Gera a tabela com `predict_period`.
5. Imprime: nome do artifact, ticker, tamanho da janela e limiar;
   período, número de janelas e de outliers (com percentual); escore
   médio; as `--top-outliers` janelas de maior escore; e a situação
   da última janela (`OUTLIER` ou `normal`).
6. Retorna a tabela.

## Erros esperados

Os erros de `WandbAutoencoderRegistry` (spec 03),
`AutoencoderPredictor` (spec 01) e `fetch_price_series`
(`training/01`) são propagados sem tratamento.

## Critérios de aceite

Use um `AutoencoderPredictor` com um `SimpleAutoencoder(input_dim=5)`
sem treino e uma série sintética de preços indexada por dias úteis.

- **CA-01:** sem argumentos, o parser deixa `ticker is None`,
  `top_outliers == 5` e `end_date` igual à data de hoje.
- **CA-02:** com 10 preços e `window_size=5`, `build_prediction_table`
  tem 5 linhas, colunas `["close_price", "reconstruction_score",
  "is_outlier"]` e índice igual às datas dos preços 5 a 9.
- **CA-03:** o `close_price` de cada linha é o preço da linha
  `window_end_index + 1` da série original.
- **CA-04:** `load_predictor(registry, None)` chama só
  `load_latest_model`, e `load_predictor(registry, "AAPL")` chama só
  `load_model("AAPL")`.
- **CA-05:** `predict_period` chama `fetch_price_series` com o ticker
  dos metadados do preditor e as datas recebidas.
- **CA-06:** `main([])`, com o registro e `fetch_price_series`
  substituídos por mocks, carrega o modelo mais recente e retorna a
  tabela de predição.
- **CA-07:** `main(["--ticker", "PETR4.SA"])` carrega o modelo de
  `PETR4.SA`.

## Fora de escopo

- Salvar a tabela em arquivo.
- Gráficos e sinais de compra e venda.
