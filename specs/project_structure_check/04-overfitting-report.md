# 04 - Relatório de overfitting

- **Arquivo:** `src/generative_models/models/<nome_do_modelo>/overfitting.py`
- **Arquivo de teste:** `tests/generative_models/models/<nome_do_modelo>/test_overfitting.py`
- **Depende de:** specs 02 e 03
- **Situação:** a implementar

## Objetivo

Gravar, depois do treino, um relatório em JSON com as perdas do modelo,
e decidir se ele passa nos critérios do job `evaluate`.

## Formato do relatório

Arquivo `<nome>-overfitting.json`, com estas chaves:

| Chave | Tipo | Significado |
|---|---|---|
| `ticker` | texto | ticker do treino |
| `model_file_path` | texto | caminho do arquivo `.pt` |
| `final_train_loss` | número | perda média da última época |
| `train_evaluation_loss` | número | erro médio quadrático no conjunto de `train`, depois do treino |
| `validation_loss` | número | erro médio quadrático no conjunto de `validation` |
| `overfitting_ratio` | número | `validation_loss / train_evaluation_loss` |
| `test_loss` | número ou `null` | erro médio quadrático no conjunto de `test` |
| `baseline_test_loss` | número ou `null` | erro médio quadrático de prever zero no `test` |

## Critérios de aprovação

Um modelo passa quando as duas condições são verdadeiras:

1. `overfitting_ratio` é menor ou igual a `1.5`;
2. `test_loss` é menor que `baseline_test_loss`.

Se o relatório não tiver `test_loss` ou `baseline_test_loss`, o modelo
não passa.

## Interface pública

```python
def main(
    command_line_arguments: list[str] | None = None,
) -> bool:
    ...
```

A linha de comando é:

```bash
uv run python -m generative_models.models.<nome>.overfitting \
    --report-path artifacts/<nome>-overfitting.json
```

## Comportamento

1. Lê o relatório indicado por `--report-path`.
2. Imprime uma linha de resumo com as perdas e o resultado.
3. Se a variável `GITHUB_STEP_SUMMARY` existir, acrescenta uma tabela em
   Markdown a esse arquivo.
4. Sai com código `0` quando o modelo passa, e com código `1` quando não passa.

## Critérios de aceite

- **CA-01:** um relatório com todas as chaves e valores válidos gera código `0`.
- **CA-02:** `overfitting_ratio` acima de `1.5` gera código `1`.
- **CA-03:** `test_loss` maior ou igual a `baseline_test_loss` gera código `1`.
- **CA-04:** um relatório sem `test_loss` gera código `1`.

## Fora de escopo

- Escolher outro limite de overfitting. O valor padrão é `1.5`, e pode
  ser alterado pelo argumento `--max-overfitting-ratio`.
