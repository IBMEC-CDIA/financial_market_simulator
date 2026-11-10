# 10 - Contrato de saída do GET /forecast

- **Arquivo:** `src/generative_models/serving/forecast.py` e rota `GET /forecast` em `src/generative_models/serving/api.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_forecast.py`, `tests/generative_models/serving/test_train_regression_forecaster.py` e `tests/generative_models/serving/test_api_forecast.py`
- **Depende de:** specs 05, 06 e 07, e de um modelo de regressão treinado por ticker
- **Situação:** implementado. O treino do artefato é feito por `serving/train_regression_forecaster.py`.

## Objetivo

Dar ao FinBrain uma previsão de um dia para o fechamento de um ticker, feita só com dados até uma data de referência. Com isso, o FinBrain pode percorrer um período dia a dia, comparando a previsão com o fechamento real do dia seguinte.

O autoencoder detecta janelas anômalas e não prevê preço. Esta rota usa outro modelo: o regressor feed-forward de `src/generative_models/models/regression/`. Ele é treinado por ticker e prevê o próximo retorno logarítmico normalizado a partir de uma janela de retornos passados.

## Origem

- Notebook `notebooks/regression_log_return_training.ipynb`: janela de 30 retornos logarítmicos, normalização com média e desvio do treino, alvo é o próximo retorno, e a reconstrução do preço é `preço_anterior * exp(retorno_previsto)`.
- Spec 08: o consumidor (FinBrain) espera o mesmo estilo de contrato, com campos fixos e erros em `detail`.

## Interface pública

```python
@dataclass(frozen=True)
class RegressionForecaster:
    ticker: str
    model: torch.nn.Module
    input_window_size: int
    return_train_mean: float
    return_train_std: float
    artifact_name: str


def load_regression_forecaster(
    ticker: str,
    model_directory: Path,
) -> RegressionForecaster | None:
    ...


def forecast_next_close(
    forecaster: RegressionForecaster,
    closes: pd.Series,
    as_of: date,
) -> dict[str, Any]:
    ...


# rota GET /forecast
def get_forecast(
    ticker: str,
    as_of: date,
) -> dict[str, Any]:
    ...
```

Parâmetros da rota (query string):

- `ticker` (obrigatório): ticker do modelo, como `PETR4.SA`, sem diferença entre maiúsculas e minúsculas.
- `as_of` (obrigatório, `YYYY-MM-DD`): último dia cujos dados o modelo pode usar. Não pode estar no futuro.

## Artefato do modelo

O modelo de cada ticker fica em `<AUTOENCODER_MODEL_DIR>/regression/<TICKER>.pt`. Quando `AUTOENCODER_MODEL_DIR` não está definida, o diretório padrão é `models/`. O arquivo é um dicionário salvo com `torch.save`, com estas chaves:

| Chave | Tipo | Conteúdo |
|---|---|---|
| `state_dict` | dict | Pesos de `FeedForwardRegressionModel`. |
| `input_window_size` | int | Tamanho da janela de entrada (30 no notebook). |
| `hidden_layer_size` | int | Unidades de cada camada oculta (64 no notebook). |
| `return_train_mean` | float | Média dos retornos logarítmicos do treino. |
| `return_train_std` | float | Desvio dos retornos logarítmicos do treino. |
| `artifact_name` | str | Nome da versão do modelo, usado em depuração. |

A carga usa `torch.load(..., weights_only=True)`. Um arquivo que não tem essas chaves é tratado como ausente, e a rota responde `404`.

## Cálculo

1. Carregar os fechamentos de `as_of - 120 dias` até `as_of`, inclusive. Nenhum dia depois de `as_of` entra no cálculo.
2. Calcular os retornos logarítmicos `ln(close_t / close_{t-1})`.
3. Pegar os últimos `input_window_size` retornos.
4. Normalizar: `(retorno - return_train_mean) / return_train_std`.
5. Rodar o modelo em modo de avaliação e sem gradiente. A saída é o retorno normalizado previsto.
6. Desnormalizar: `retorno_previsto = saída * return_train_std + return_train_mean`.
7. `predicted_close = close_as_of * exp(retorno_previsto)`, arredondado para 4 casas.
8. `target_date` é o próximo dia útil (segunda a sexta) depois de `as_of`. Feriados não são considerados nesta versão.

Se houver menos de `input_window_size + 1` fechamentos até `as_of`, o cálculo não é feito e a rota responde `422`.

## Formato da resposta

`GET /forecast` responde `200` com um objeto JSON com exatamente estes campos:

```json
{
  "ticker": "PETR4.SA",
  "as_of": "2026-05-12",
  "target_date": "2026-05-13",
  "predicted_close": 36.1235,
  "window_days": 30
}
```

| Campo | Tipo JSON | Regra |
|---|---|---|
| `ticker` | string | Ticker pedido, em maiúsculas. |
| `as_of` | string | Data de referência, `YYYY-MM-DD`. |
| `target_date` | string | Próximo dia útil depois de `as_of`, `YYYY-MM-DD`. |
| `predicted_close` | number | Preço previsto, finito e maior que zero. |
| `window_days` | integer | Tamanho da janela de entrada (`input_window_size`). |

## Erros

| Status | Quando | Corpo |
|---|---|---|
| `200` | Previsão calculada. | Objeto acima. |
| `404` | Não há artefato de regressão para o ticker. | `{"detail": "No forecast model for ticker 'X'."}` |
| `422` | `as_of` no futuro, ou menos de `input_window_size + 1` fechamentos até `as_of`. | `{"detail": "<texto>"}` |

O `404` é a resposta normal para um ticker sem modelo de regressão treinado. O FinBrain trata esse caso como "o modelo não cobre este ativo".

## Comportamento

- A rota não usa dados depois de `as_of`, mesmo que o yfinance os tenha. Isso permite repetir a previsão dia a dia sem vazamento de informação.
- A rota não faz treino. Se não houver artefato, a resposta é `404`.
- O artefato é lido a cada requisição. Assim, um modelo treinado de novo entra na resposta sem reiniciar o serviço.
- Esta rota não recomenda compra ou venda. A previsão é uma estimativa numérica.

## Treino do artefato

`python -m generative_models.serving.train_regression_forecaster --ticker AAPL`
baixa os preços desde 2018 até hoje, treina o regressor e salva
`models/regression/AAPL.pt` (ou `<AUTOENCODER_MODEL_DIR>/regression/`). O treino:

- usa as primeiras 80% dos retornos para treino e o resto para validação;
- normaliza com média e desvio apenas do treino;
- usa Adam com taxa 0,001, lotes de 32 e até 100 épocas;
- guarda os pesos da época de menor perda de validação (parada antecipada);
- registra a execução com `WandbExperimentTracker`.

Resultado atual com os tickers do deploy: a perda de validação ficou um pouco
acima da linha de base (prever a média do treino) em AAPL, GOOGL e MSFT. A spec
exige que o modelo supere a linha de base. Esse critério não foi atingido, e a
previsão deve ser tratada como estimativa.

## Fora do escopo

- Feriados de bolsa no `target_date`, que hoje são ignorados.
- Previsão para mais de um dia.
- Treino com dados apenas anteriores a cada data de uma reprodução histórica.
  Os artefatos de hoje usam dados até hoje, então a previsão de datas passadas
  não é uma avaliação fora da amostra.

## Definição de pronto

- [x] `forecast.py` tem as funções da interface, com docstrings completas.
- [x] Testes cobrem o cálculo com um modelo pequeno e pesos fixos, a desnormalização, o próximo dia útil, a ausência de artefato, `as_of` no futuro e dados insuficientes.
- [x] Teste de rota cobre `200`, `404`, `422` e o fato de que nenhum dado depois de `as_of` é usado.
- [x] `specs/serving/README.md` e o README raiz listam a rota.
- [ ] O modelo supera a linha de base de validação (prever a média do treino). Não atingido com os artefatos atuais.
