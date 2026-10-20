# 08 - Script de linha de comando

- **Arquivo:** `src/generative_models/models/autoencoder/train_autoencoder.py`
- **Arquivo de teste:** `tests/generative_models/models/autoencoder/test_train_autoencoder.py`
- **Depende de:** specs 05 e 07
- **Situação:** a implementar

## Objetivo

Permitir executar o treinamento completo pelo terminal, sem abrir o
notebook, alterando qualquer hiperparâmetro por argumento:

```bash
uv run python -m generative_models.models.autoencoder.train_autoencoder \
    --ticker AAPL --epochs 100
```

## Origem no notebook

Não existe uma célula equivalente: este módulo substitui o ato de
"editar as constantes e rodar todas as células".

## Interface pública

```python
def build_argument_parser() -> argparse.ArgumentParser:
    ...


def build_training_config(
    arguments: argparse.Namespace,
) -> AutoencoderTrainingConfig:
    ...


def main(
    command_line_arguments: list[str] | None = None,
) -> AutoencoderTrainingResult:
    ...
```

O módulo termina com `if __name__ == "__main__": main()`.

## Comportamento

### `build_argument_parser`

Cria um `argparse.ArgumentParser` com os argumentos abaixo. Os
valores padrão são **lidos das dataclasses da spec 05** (por exemplo,
`MarketDataConfig().window_size`) e nunca repetidos como literais no
script.

| Argumento | Tipo | Campo da configuração |
|---|---|---|
| `--ticker` (obrigatório) | `str` | passado ao treinador |
| `--start-date` | `str` | `market_data.start_date` |
| `--end-date` | `str` | `market_data.end_date` |
| `--train-split-ratio` | `float` | `market_data.train_split_ratio` |
| `--window-size` | `int` | `market_data.window_size` |
| `--batch-size` | `int` | `market_data.batch_size` |
| `--hidden-layer-size` | `int` | `architecture.hidden_layer_size` |
| `--latent-layer-size` | `int` | `architecture.latent_layer_size` |
| `--epochs` | `int` | `optimization.number_of_epochs` |
| `--learning-rate` | `float` | `optimization.learning_rate` |
| `--random-seed` | `int` | `optimization.random_seed` |
| `--contamination` | `float` | `contamination` |
| `--project-name` | `str` | `tracking.project_name` |
| `--output-directory` | `str` | `tracking.output_directory` |
| `--env-file-path` | `str` | `tracking.env_file_path` |
| `--device` | `str` | `device` |

### `build_training_config`

Monta um `AutoencoderTrainingConfig` a partir do `Namespace`,
preenchendo cada campo conforme a tabela acima. Valores inválidos
geram o `ValueError` da validação da spec 05.

### `main`

1. Faz o parse de `command_line_arguments` (ou de `sys.argv` quando
   for `None`).
2. Monta a configuração com `build_training_config`.
3. Cria `AutoencoderTrainer(ticker=<--ticker>, config=<config>)` e
   chama `train()`.
4. Imprime uma linha com o ticker, a última perda de treino, a perda
   de validação, o limiar de anomalia e o caminho do modelo salvo.
5. Retorna o `AutoencoderTrainingResult`.

`AutoencoderTrainer` deve ser importado no topo do módulo, para que
os testes possam substituí-lo com
`patch("generative_models.models.autoencoder.train_autoencoder.AutoencoderTrainer")`.

## Erros esperados

| Situação | Exceção |
|---|---|
| `--ticker` ausente | `SystemExit` (gerado pelo `argparse`) |
| Valor inválido para a configuração (por exemplo, `--epochs 0`) | `ValueError` (spec 05) |

## Critérios de aceite

- **CA-01:** fazer o parse de `[]` (sem `--ticker`) lança
  `SystemExit`.
- **CA-02:** com apenas `["--ticker", "AAPL"]`, a configuração gerada
  é igual a `AutoencoderTrainingConfig()` (todos os padrões).
- **CA-03:** com `--epochs 50 --window-size 20 --learning-rate 0.01
  --device cpu`, a configuração tem `number_of_epochs == 50`,
  `window_size == 20`, `learning_rate == 0.01` e `device == "cpu"`.
- **CA-04:** `--train-split-ratio` e `--contamination` são
  convertidos para `float`, e `--batch-size` para `int`.
- **CA-05:** `main(["--ticker", "AAPL", "--epochs", "2"])`, com o
  treinador substituído por um mock, cria o treinador com
  `ticker="AAPL"` e uma configuração com `number_of_epochs == 2`,
  chama `train()` uma vez e retorna o resultado do mock.
- **CA-06:** `main(["--ticker", "AAPL", "--epochs", "0"])` lança
  `ValueError` antes de criar o treinador.

## Fora de escopo

- Treinar vários tickers em uma única execução.
- Leitura de argumentos a partir de arquivo de configuração.
