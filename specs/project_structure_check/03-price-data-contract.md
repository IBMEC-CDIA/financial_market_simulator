# 03 - Contrato dos dados de preços

- **Arquivo:** `src/generative_models/data/collect_market_data.py` (produção)
  e `src/generative_models/data/validation.py` (checagens)
- **Arquivo de teste:** `tests/generative_models/data/test_collect_market_data.py`
  e `tests/generative_models/data/test_validation.py`
- **Depende de:** nenhuma
- **Situação:** fornecido pelo projeto. Os modelos novos apenas consomem
  este CSV.

## Objetivo

Garantir que todos os modelos recebam os mesmos dados, já validados e
divididos da mesma forma. Por isso a coleta e a validação não são
repetidas dentro de cada modelo.

## Formato do CSV

Arquivo com cabeçalho e índice de data:

| Coluna | Tipo | Conteúdo |
|---|---|---|
| `date` | data | índice, em ordem crescente e sem repetição |
| `close_price` | número | preço de fechamento, sempre maior que zero |
| `log_return` | número | log-retorno diário; vazio na primeira linha |
| `split` | texto | `train`, `validation` ou `test`; vazio na primeira linha |

Os rótulos de `split` seguem a ordem do tempo: primeiro todos os `train`,
depois os `validation` e por último os `test`.

## Checagens de qualidade

O CSV só é gravado se a validação passar. As regras estão em
`validation.py`:

- pelo menos `MINIMUM_ROW_COUNT` linhas (252);
- `close_price` sem valores nulos e estritamente positivo;
- `date` única e estritamente crescente;
- `log_return` entre `-MAXIMUM_ABS_LOG_RETURN` e `+MAXIMUM_ABS_LOG_RETURN` (0,5).

## Regras para o modelo

- O modelo lê `log_return` e `split` do CSV. Ele não baixa preços por conta
  própria.
- O modelo não altera o conjunto de teste durante o treino. O conjunto de
  teste é usado uma única vez, na avaliação.
- A normalização usa apenas as estatísticas do conjunto de `train`.

## Critérios de aceite

- **CA-01:** um CSV com `split` ausente é rejeitado pelo treino com erro
  claro.
- **CA-02:** um CSV com preço zero ou negativo não é gravado pelo
  `collect_market_data`.
- **CA-03:** o modelo novo usa `split` do CSV e não recalcula a divisão.

## Fora de escopo

- Escolher outros ativos ou outro período (isso é feito pelos argumentos
  do workflow).
