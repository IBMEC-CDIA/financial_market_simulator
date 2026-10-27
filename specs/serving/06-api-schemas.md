# 06 - Schemas da API

- **Arquivo:** `src/generative_models/serving/api_schemas.py`
- **Arquivo de teste:** coberto pelos testes da API, em
  `tests/generative_models/serving/test_api.py` (spec 07). Um arquivo
  `test_api_schemas.py` é opcional para os critérios abaixo.
- **Depende de:** nenhuma
- **Situação:** a implementar

## Objetivo

Definir, com `pydantic`, o formato de todas as requisições e
respostas da API. As validações declaradas aqui (por exemplo, preços
positivos) são aplicadas pelo FastAPI antes de o código da rota rodar
e aparecem automaticamente na documentação Swagger.

## Origem

Não existe no notebook. Os campos espelham os dataclasses
`AutoencoderModelMetadata` e `AutoencoderPrediction` (spec 01) e
`AvailableModel` (spec 05).

## Interface pública

Todas as classes herdam de `pydantic.BaseModel`. Todo campo deve ter
`Field(description=..., examples=[...])`, para a documentação
OpenAPI.

```python
TickerField = Annotated[str | None, Field(default=None, ...)]


class HealthResponse(BaseModel):
    status: str
    default_ticker: str | None
    loaded_tickers: list[str]


class ModelMetadataResponse(BaseModel):
    ticker: str
    artifact_name: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float


class AvailableModelResponse(BaseModel):
    artifact_name: str
    ticker: str
    created_at: str


class ReloadRequest(BaseModel):
    ticker: TickerField


class LogReturnPredictionRequest(BaseModel):
    ticker: TickerField
    log_returns: list[float]


class PricePredictionRequest(BaseModel):
    ticker: TickerField
    prices: list[Annotated[float, Field(gt=0)]]


class WindowPredictionResponse(BaseModel):
    window_end_index: int
    window_end_date: date | None = None
    close_price: float | None = None
    reconstruction_score: float
    is_outlier: bool


class PredictionResponse(BaseModel):
    model: ModelMetadataResponse
    number_of_windows: int
    number_of_outliers: int
    outlier_rate: float
    windows: list[WindowPredictionResponse]


class ErrorResponse(BaseModel):
    detail: str
```

## Comportamento

- `TickerField` é opcional, com padrão `None`: sem ticker, a API usa o
  modelo padrão.
- `LogReturnPredictionRequest.log_returns` exige **pelo menos 1**
  valor (`min_length=1`).
- `PricePredictionRequest.prices` exige **pelo menos 2** valores
  (`min_length=2`), todos **estritamente positivos** (`gt=0`).
- O tamanho mínimo real (`window_size` ou `window_size + 1`) depende
  do modelo e é validado pelo preditor (spec 01), não pelo schema.
- `ReloadRequest`, `LogReturnPredictionRequest` e
  `PricePredictionRequest` declaram exemplos de corpo completos em
  `model_config = ConfigDict(json_schema_extra={"examples": [...]})`,
  para o botão "Try it out" do Swagger.
- `window_end_date` e `close_price` só são preenchidos pela rota de
  dados de mercado; nas demais rotas ficam `null`.
- `ErrorResponse` é o formato de todos os erros documentados
  (`{"detail": "..."}`).

## Erros esperados

| Situação | Exceção |
|---|---|
| Corpo que viola as restrições acima | `pydantic.ValidationError` (vira HTTP 422 no FastAPI) |

## Critérios de aceite

- **CA-01:** `LogReturnPredictionRequest(log_returns=[])` lança
  `ValidationError`, e com `[0.01]` é válido com `ticker is None`.
- **CA-02:** `PricePredictionRequest(prices=[10.0])` lança
  `ValidationError` (menos de 2 preços).
- **CA-03:** `PricePredictionRequest(prices=[10.0, 0.0, 11.0])` e
  `PricePredictionRequest(prices=[10.0, -1.0])` lançam
  `ValidationError`.
- **CA-04:** `WindowPredictionResponse` criado sem `window_end_date`
  e `close_price` serializa os dois como `None`.
- **CA-05:** `ReloadRequest()` é válido, com `ticker is None`.
- **CA-06:** o JSON schema de `PricePredictionRequest`
  (`model_json_schema()`) contém `examples`, e todos os campos de
  `ModelMetadataResponse` têm `description`.

## Fora de escopo

- Validar o ticker contra a lista de modelos disponíveis (feito pela
  API ao carregar o modelo).
