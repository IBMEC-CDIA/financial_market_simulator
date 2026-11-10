# 09 - Download dos modelos mais recentes

- **Arquivo:** `src/generative_models/serving/download_latest_models.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_download_latest_models.py`
- **Depende de:** specs 03 e 04, `training/07`
- **Situação:** implementado

## Objetivo

Preparar a pasta `models/` com a versão de cada modelo registrado no wandb sob o alias do experimento, para que o build da imagem funcione sem um run de treino. Quem usa é o job `build-image` do `build.yml` quando o input `train_run_id` está vazio.

## Origem

- Spec 03: o registro no wandb já sabe montar o nome do projeto e acessar o cliente (`project_path` e `api`). O script reusa essas propriedades e não carrega os modelos em memória.
- `training/07`: o nome do artifact (`simple-autoencoder-<ticker_slug>`) e os metadados gravados pelo trainer.
- Spec 04 do `project_structure_check`: o formato do relatório de overfitting, que o registro local lê (`<nome>.pt` e `<nome>-overfitting.json`).

## Interface pública

```python
REPORT_SUFFIX = "-overfitting.json"
DEFAULT_OUTPUT_DIRECTORY = "models"


def build_argument_parser() -> argparse.ArgumentParser:
    ...


def parse_ticker_list(ticker_text: str) -> list[str]:
    ...


def build_model_name(ticker: str) -> str:
    ...


def build_report_data(
    metadata: dict,
    model_file_name: str,
) -> dict:
    ...


def download_latest_model(
    registry: WandbAutoencoderRegistry,
    ticker: str,
    output_directory: Path,
) -> Path:
    ...


def main(
    command_line_arguments: list[str] | None = None,
) -> list[Path]:
    ...
```

Linha de comando:

```bash
uv run python -m generative_models.serving.download_latest_models \
  --tickers PETR4.SA,AAPL \
  --output-directory models
```

| Opção | Obrigatória | Padrão | Uso |
|---|---|---|---|
| `--tickers` | sim | - | Tickers separados por vírgula. Espaços e itens vazios são ignorados. |
| `--output-directory` | não | `models` | Pasta onde os arquivos são gravados. É criada se não existir. |
| `--project-name` | não | `TrackingConfig.project_name` | Projeto do wandb onde estão os modelos. |
| `--entity` | não | conta padrão do wandb | Usuário ou time do wandb. |

A chave do wandb vem da variável `WANDB_API_KEY`, pelo próprio cliente do wandb. No `build.yml` ela vem apenas do secret `WANDB_API_KEY`.

## Comportamento

1. `main` lê as opções, separa os tickers e cria a pasta de saída. A validação dos tickers acontece antes de qualquer chamada ao wandb.
2. Para cada ticker, `download_latest_model` busca o artifact `<projeto>/simple-autoencoder-<ticker_slug>:<alias>` do tipo `model`. O alias padrão é `latest`, e o build passa o nome do experimento.
3. Antes do download, confere se os metadados do artifact têm as chaves de `REQUIRED_METADATA_KEYS`. Se faltar alguma, o script para com erro.
4. O artifact é baixado para uma pasta temporária. O primeiro arquivo `*.pt` é copiado para `<output>/simple-autoencoder-<ticker_slug>.pt`.
5. `build_report_data` monta o relatório com os valores do artifact e grava `<output>/simple-autoencoder-<ticker_slug>-overfitting.json`, em JSON com recuo de 2 espaços.
6. Os campos do relatório são:
   - `ticker`, `window_size`, `hidden_layer_size`, `latent_layer_size`, `return_train_mean`, `return_train_std` e `anomaly_threshold` vêm dos metadados.
   - `final_train_loss`, `train_evaluation_loss` e `validation_loss` vêm dos metadados, quando existem.
   - `overfitting_ratio` é `validation_loss / train_evaluation_loss`. Fica `null` se a perda de treino for zero ou ausente.
   - `test_loss` e `baseline_test_loss` ficam sempre `null`, porque o trainer não os grava no artifact.
   - `model_file_path` recebe o nome do arquivo `.pt`.
7. `main` imprime uma linha por ticker e devolve a lista dos caminhos dos relatórios, na ordem dos tickers.

O JSON gravado é lido pelo `load_overfitting_report` (spec 04 do `project_structure_check`). Por isso o relatório precisa ter todos os campos de `OverfittingReport`, mesmo que alguns sejam `null`.

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| `--tickers` vazio ou só com separadores | `ValueError` | `At least one ticker is required.` |
| Artifact com o alias inexistente no wandb | erro do cliente do wandb (`CommError`) | nome do artifact |
| Metadados sem chave necessária | `ValueError` | `is missing metadata required for inference: <chaves>.` |
| Artifact sem arquivo `.pt` | `FileNotFoundError` | `has no .pt weights file.` |
| Opção desconhecida ou ausente | `SystemExit` (argparse) | mensagem do argparse |

Em todos os casos de erro nenhum relatório é gravado para o ticker com problema. Os tickers já processados antes do erro permanecem na pasta.

## Integração com o build

- O job `build-image` roda o script só quando `train_run_id` está vazio. Com `train_run_id`, ele baixa o artifact `trained-models` do run informado, como antes.
- A chave do wandb é mascarada no log antes da etapa de checagem das entradas.
- A checagem das entradas falha o job antes do download se faltar `tickers` ou a chave do wandb.

## Critérios de aceite

Cada critério deve virar ao menos um teste.

- **CA-01:** `parse_ticker_list` mantém a ordem, ignora espaços e itens vazios, e recusa uma lista vazia. (`test_parse_ticker_list_*`)
- **CA-02:** o nome do modelo segue o slug do trainer, por exemplo `simple-autoencoder-petr4.sa`. (`test_build_model_name_uses_the_trainer_slug`)
- **CA-03:** o parser usa `models` como pasta padrão e `entity` vazio por padrão. (`test_build_argument_parser_defaults_to_the_models_directory`)
- **CA-04:** `overfitting_ratio` é a razão entre a validação e o treino, e vira `null` quando o treino é zero. (`test_build_report_data_*`)
- **CA-05:** o download grava `<nome>.pt` com os bytes do artifact e um relatório que o `load_overfitting_report` lê. (`test_download_latest_model_writes_weights_and_report`)
- **CA-06:** um artifact sem uma chave de serving é recusado, e nenhum arquivo é gravado. (`test_download_latest_model_refuses_incomplete_metadata`)
- **CA-07:** um artifact sem `.pt` é recusado. (`test_download_latest_model_refuses_an_artifact_without_weights`)
- **CA-08:** `main` baixa todos os tickers informados e imprime uma linha por ticker. (`test_main_downloads_every_ticker`)

## Fora de escopo

- Não baixa o `.pt` de um run específico. Para isso, o build usa `train_run_id`.
- Não grava `test_loss` nem `baseline_test_loss`. Gravar esses valores exige que o trainer os registre no artifact, o que é uma mudança separada.
- Não carrega os modelos em memória. A checagem de compatibilidade fica para o serving.
- Não tenta novamente um download que falhou.
