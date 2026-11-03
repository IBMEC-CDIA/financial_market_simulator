# 02 - Contrato de treino

- **Arquivo:** `src/generative_models/models/<nome_do_modelo>/run_train_script.py`
- **Arquivo de teste:** `tests/generative_models/models/<nome_do_modelo>/test_run_train_script.py`
- **Depende de:** specs 01, 03 e 04
- **Situação:** a implementar

## Objetivo

Treinar o modelo para um ticker a partir da linha de comando, gravar o
modelo e o relatório de overfitting, e devolver o controle ao workflow
`train.yml`.

## Interface pública

```python
def build_argument_parser() -> argparse.ArgumentParser:
    ...


def main(
    command_line_arguments: list[str] | None = None,
) -> AutoencoderTrainingResult:
    ...
```

O nome da função de treino pode mudar, mas o comportamento de linha de
comando abaixo é obrigatório.

## Argumentos obrigatórios

| Argumento | Obrigatório | Uso no `train.yml` |
|---|---|---|
| `--ticker` | sim | ticker do treino, por exemplo `AAPL` |
| `--price-data-path` | sim, no uso local | CSV da spec 03. O workflow usa o `run_model_training` (spec 06), que baixa os preços pelo manager. |
| `--epochs` | não | número de épocas |
| `--learning-rate` | não | taxa de aprendizado |
| `--window-size` | não | tamanho da janela |
| `--device` | não | `cpu` ou `cuda` |
| `--output-directory` | não | pasta onde os arquivos são gravados |

Os demais argumentos (`--start-date`, `--end-date`, `--batch-size`,
`--random-seed` e outros) devem ter valor padrão, para o comando rodar
só com `--ticker`.

## Comportamento

1. Lê o CSV de `--price-data-path` e usa a coluna `split` para separar
   treino, validação e teste (spec 03).
2. Treina o modelo com o `WandbExperimentTracker`. Com `WANDB_MODE=disabled`,
   o treino roda sem chave e sem enviar nada.
3. Grava, na pasta `--output-directory`:
   - `<nome>.pt`, com o `state_dict` do modelo;
   - `<nome>-overfitting.json`, com o relatório da spec 04.
4. Imprime uma linha de resumo com o ticker e as perdas.
5. Sai com código `0` em caso de sucesso.

## Erros esperados

| Situação | Comportamento |
|---|---|
| `--ticker` ausente | `argparse` sai com código diferente de zero |
| CSV sem coluna `close_price` ou `split` | `ValueError`, e a execução para |
| Dados insuficientes para uma das janelas | `ValueError`, e a execução para |
| Erro de rede ao falar com o wandb | o treino continua com o tracker desativado |

## Critérios de aceite

- **CA-01:** `python -m generative_models.models.<nome>.run_train_script --help`
  mostra todos os argumentos da tabela.
- **CA-02:** com `--price-data-path` apontando para um CSV válido e
  `--epochs 2`, o comando termina com código `0` e grava os dois arquivos.
- **CA-03:** com `WANDB_MODE=disabled` e sem `WANDB_API_KEY`, o comando
  termina com código `0`.
- **CA-04:** o nome do arquivo `.pt` e do relatório derivam do nome do
  modelo e do ticker, sem espaços.

## Fora de escopo

- Baixar preços. Este comando usa o CSV pronto. Para vários tickers, o
  download é feito pelo manager (spec 06).
- Avaliar o modelo (isso é feito pelo job `evaluate`, spec 04).
