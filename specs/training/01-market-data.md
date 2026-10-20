# 01 - Dados de mercado

- **Arquivo:** `src/generative_models/data/market_data.py`
- **Arquivo de teste:** `tests/generative_models/data/test_market_data.py`
- **Depende de:** nenhuma

## Objetivo

Isolar a coleta de preços históricos (via `yfinance`) e o cálculo de
log-retornos em funções reutilizáveis, para que o restante do pacote
não dependa diretamente da API do `yfinance`.

## Origem no notebook

- Célula 12: `fetch_price_series(TICKER, START_DATE, END_DATE)`. No
  notebook a função vem pronta do pacote; aqui ela passa a ser
  implementada por você.
- Célula 16: cálculo do log-retorno
  `log(price[t + 1]) - log(price[t])`, aqui em uma versão baseada em
  `numpy`.

## Interface pública

```python
def fetch_price_series(
    ticker_symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    ...


def compute_log_return_values(
    price_values: np.ndarray,
) -> np.ndarray:
    ...
```

## Comportamento

### `fetch_price_series`

- Chama `yfinance.download(ticker_symbol, start=start_date,
  end=end_date, progress=True)`.
- Mantém apenas a coluna `Close` do resultado e remove as linhas com
  valores ausentes.
- Retorna um `DataFrame` com **uma única coluna**, chamada
  `close_price`, indexado pelas datas devolvidas pelo `yfinance`.

### `compute_log_return_values`

- Recebe um array 1D com `n` preços consecutivos e positivos.
- Retorna um array 1D com `n - 1` elementos, em que o elemento `i` é
  `log(price_values[i + 1]) - log(price_values[i])`.
- O array retornado tem `dtype` `float32`.

## Erros esperados

Nenhum erro próprio. Erros do `yfinance` (rede, ticker inválido) são
propagados sem tratamento.

## Critérios de aceite

- **CA-01:** `compute_log_return_values(np.array([100.0, 110.0]))`
  retorna um array com um único valor, igual a `log(110 / 100)`
  (tolerância de `1e-6`).
- **CA-02:** para uma entrada com `n` preços, o resultado tem `n - 1`
  elementos.
- **CA-03:** o resultado de `compute_log_return_values` tem `dtype`
  `np.float32`.
- **CA-04:** `fetch_price_series` chama `yfinance.download` com o
  ticker, `start=start_date`, `end=end_date` e `progress=True`
  (verifique com um mock de `yfinance.download`).
- **CA-05:** `fetch_price_series` retorna apenas a coluna
  `close_price`, mesmo quando o `DataFrame` devolvido pelo mock tem
  outras colunas (`Open`, `High`, `Low`, `Volume`).
- **CA-06:** linhas com `Close` ausente (`NaN`) são removidas do
  resultado.

## Fora de escopo

- Cache local dos dados baixados.
- Outras frequências além da diária e outras colunas além do
  fechamento.
- Validação do formato das datas.
