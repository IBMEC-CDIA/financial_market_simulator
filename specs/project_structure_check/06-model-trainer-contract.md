# 06 - Contrato de treinador e gerenciador de tickers

- **Arquivo:** `src/generative_models/models/ticker_model_manager.py`
  (gerenciador) e `src/generative_models/models/<nome_do_modelo>/trainer_factory.py`
  (contrato de cada modelo)
- **Arquivo de teste:** `tests/generative_models/models/test_ticker_model_manager.py`
  e `tests/generative_models/models/<nome_do_modelo>/test_trainer_factory.py`
- **Depende de:** specs 01, 03 e 04
- **Situação:** a implementar

## Objetivo

Permitir que um único gerenciador treine o modelo de vários tickers, sem
conhecer a arquitetura de cada modelo. Cada pacote de modelo expõe uma
fábrica de treinadores, e o gerenciador só usa a interface descrita aqui.

## Contrato de cada modelo: `trainer_factory.py`

Todo pacote em `models/` tem o módulo `trainer_factory.py` com:

```python
def build_trainer(
    ticker: str,
    price_data: pd.DataFrame | None = None,
    options: TrainingOptions | None = None,
) -> Trainer:
    ...
```

- `ticker`: símbolo do ativo.
- `price_data`: DataFrame no formato da spec 03 (`close_price`,
  `log_return` e `split`). Quando `None`, o treinador baixa os preços pela
  configuração própria do modelo.
- `options`: opções de treino compartilhadas (`TrainingOptions`, em
  `models/training_options.py`). Quando `None`, o modelo usa a configuração
  padrão dele.
- `Trainer` é um objeto com o método `train()`, que treina, grava os
  arquivos da spec 02 e devolve um resultado.

O resultado de `train()` tem, no mínimo, os atributos `model_file_path`,
`validation_loss`, `train_evaluation_loss` e `anomaly_threshold`. Quando o
modelo tem conjunto de teste, também tem `test_loss` e
`baseline_test_loss`.

## Interface pública do gerenciador

```python
class TickerModelManager:
    def __init__(
        self,
        tickers: dict[str, dict[str, str]],
        trainer_factory: Callable[[str, pd.DataFrame], Trainer] | None = None,
    ) -> None: ...

    def list_tickers(self) -> list[str]: ...
    def get_date_range(self, ticker: str) -> tuple[str, str]: ...
    def fetch_price_data(self, ticker: str) -> pd.DataFrame: ...
    def train_model(self, ticker: str) -> Any: ...
    def train_all_models(self) -> None: ...
    def get_model(self, ticker: str) -> Any: ...
```

Quando `trainer_factory` é `None`, o gerenciador usa a fábrica do
`base_autoencoder` (`build_trainer`).

## Comando de treino por tickers

O comando compartilhado `models/run_model_training.py` treina vários tickers
de um modelo, escolhido com `--model`. Ele não é copiado para cada pacote:

```bash
uv run python -m generative_models.models.run_model_training     --model base_autoencoder --tickers AAPL,MSFT --epochs 300     --learning-rate 0.001 --window-size 10
```

Os argumentos são `--model`, `--tickers`, `--start-date`, `--end-date`,
`--epochs`, `--learning-rate`, `--window-size`, `--output-directory` e
`--device`. O comando importa `trainer_factory.build_trainer` e
`overfitting` do pacote escolhido, e grava, para cada ticker, o relatório
`<nome>-overfitting.json` da spec 04. O `train.yml` chama esse comando com
o input `model`, e o job `evaluate` lê todos os relatórios.

## Comportamento

1. `list_tickers` devolve as chaves de `tickers`, na ordem recebida.
2. `get_date_range` devolve `(start_date, end_date)` do ticker. Um ticker
   desconhecido gera `KeyError`.
3. `fetch_price_data(ticker)` baixa os preços entre as datas do ticker, valida
   a série (spec 03), adiciona `log_return` e `split` com as proporções de
   `MarketDataConfig`, e devolve o DataFrame. O mesmo caminho é usado pelo job
   `collect-data`, para que os dados sejam iguais.
4. `train_model(ticker)` chama `fetch_price_data`, cria o treinador com
   `trainer_factory(ticker, price_data)`, executa `train()` e guarda o
   resultado em `models[ticker]`. Devolve o resultado.
5. `train_all_models()` chama `train_model` para cada ticker, em sequência e
   na ordem de `list_tickers`. Um erro interrompe o loop e é propagado.
6. `get_model(ticker)` devolve o resultado guardado. Se o ticker ainda não foi
   treinado, gera `KeyError` com a mensagem "call train_model first".

## Erros esperados

| Situação | Exceção |
|---|---|
| Ticker desconhecido em `get_date_range`, `fetch_price_data`, `train_model` ou `get_model` | `KeyError` |
| Ticker sem modelo treinado em `get_model` | `KeyError` |
| Série de preços inválida (spec 03) | `PriceValidationError` |
| Falha dentro de `train()` | propagada sem alteração |

## Critérios de aceite

- **CA-01:** `list_tickers` devolve as chaves na mesma ordem do construtor.
- **CA-02:** `get_date_range("AAPL")` devolve as datas configuradas.
- **CA-03:** `fetch_price_data` chama a coleta com as datas do ticker e
  devolve um DataFrame com `close_price`, `log_return` e `split`.
- **CA-04:** `train_model` cria o treinador pela fábrica, chama `train()` e
  guarda o resultado; `get_model` devolve esse mesmo objeto.
- **CA-05:** `train_all_models` treina na ordem da lista e não começa o
  próximo ticker antes de o anterior terminar.
- **CA-06:** `train_all_models` para no primeiro erro e não treina os
  demais tickers.
- **CA-07:** `get_model` de um ticker não treinado gera `KeyError`.
- **CA-08:** `build_trainer` do `base_autoencoder` devolve um
  `AutoencoderTrainer` configurado com o ticker e com `price_data`.

## Fora de escopo

- Escolher hiperparâmetros por ticker.
- Salvar os modelos em disco ou no wandb. Isso é feito pelo próprio
  treinador, conforme a spec 02.
- Rodar os treinos em paralelo.
