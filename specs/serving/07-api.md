# 07 - API REST

- **Arquivo:** `src/generative_models/serving/api.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_api.py`
- **Depende de:** specs 01, 04, 05 e 06

## Objetivo

Expor os autoencoders treinados por HTTP com FastAPI: consultar os
modelos em memória e no wandb, recarregar um modelo sem reiniciar o
serviço e detectar outliers em log-retornos, preços ou dados de
mercado baixados na hora.

```bash
uv run uvicorn generative_models.serving.api:app --reload
uv run python -m generative_models.serving.api
```

Documentação interativa: `http://127.0.0.1:8000/docs` (Swagger) e
`/redoc`.

## Origem

- Spec 05: o serviço singleton é a única fonte de modelos das rotas.
- Spec 04: a rota de mercado reutiliza `predict_period`.
- Spec 06: todos os corpos e respostas.

## Interface pública

```python
app: FastAPI


def get_model_service() -> AutoencoderModelService:
    ...


def build_model_metadata_response(
    metadata: AutoencoderModelMetadata,
) -> ModelMetadataResponse:
    ...


def build_prediction_response(
    metadata: AutoencoderModelMetadata,
    prediction: AutoencoderPrediction,
    prediction_table: pd.DataFrame | None = None,
) -> PredictionResponse:
    ...


def main(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    ...
```

- `get_model_service` retorna `AutoencoderModelService.get_instance()`
  e é injetada em **todas** as rotas com `Depends`. Nos testes, ela é
  substituída com `app.dependency_overrides[get_model_service]`.
- `predict_period` deve ser importada no topo do módulo a partir de
  `generative_models.serving.predict_autoencoder`.
- `main` sobe a aplicação com `uvicorn.run(app, host=host,
  port=port)`. O módulo termina com
  `if __name__ == "__main__": main()`.

## Comportamento

### Aplicação

- `FastAPI` com `title`, `summary`, `description` (como a API
  funciona, como o modelo é escolhido e a tabela de variáveis de
  ambiente), `version="0.1.0"` e três tags: `Health`, `Models` e
  `Predictions`.
- Toda rota tem `tags`, `summary`, `response_model` e docstring. As
  rotas que podem falhar documentam as respostas de erro com
  `responses={...: {"model": ErrorResponse, ...}}`.

### Inicialização (`lifespan`)

- Ao subir, carrega o modelo padrão (`model_service.get_predictor()`)
  usando o serviço de `dependency_overrides`, se houver, ou
  `get_model_service`.
- Se `AUTOENCODER_PRELOAD_MODEL` for `"false"` (sem diferenciar
  maiúsculas), não carrega nada.
- Uma falha no carregamento (por exemplo, sem internet) é registrada
  com `logger.warning` e **não** impede a API de subir.

### Rotas

| Método | Rota | Tag | Comportamento |
|---|---|---|---|
| `GET` | `/health` | Health | `HealthResponse` com `status="ok"`, `default_ticker` do serviço e os tickers de `list_loaded_models()`. Não acessa o wandb. |
| `GET` | `/models/loaded` | Models | Lista de `ModelMetadataResponse` dos modelos em memória. |
| `GET` | `/models/available` | Models | Lista de `AvailableModelResponse` de `list_available_models()`. |
| `GET` | `/models/default` | Models | `ModelMetadataResponse` de `get_predictor()`. |
| `POST` | `/models/reload` | Models | Corpo `ReloadRequest`. Retorna os metadados de `reload(ticker)`. |
| `POST` | `/predictions/log-returns` | Predictions | Corpo `LogReturnPredictionRequest`. Usa `get_predictor(ticker).predict(log_returns)`. |
| `POST` | `/predictions/prices` | Predictions | Corpo `PricePredictionRequest`. Usa `get_predictor(ticker).predict_from_prices(prices)`. |
| `GET` | `/predictions/market` | Predictions | Query `ticker`, `start_date` e `end_date` (opcionais). Detalhes abaixo. |

### `/predictions/market`

1. `end_date` padrão: hoje. `start_date` padrão: `end_date` menos 365
   dias.
2. Se `start_date >= end_date`, lança
   `ValueError("start_date must be earlier than end_date.")` (vira
   422).
3. Obtém o preditor com `get_predictor(ticker)` e chama
   `predict_period(predictor, start_date.isoformat(),
   end_date.isoformat())`.
4. Monta a resposta com `build_prediction_response`, passando a
   tabela para preencher `window_end_date` e `close_price` de cada
   janela. Os `window_end_index` começam em `window_size - 1`.

### `build_prediction_response`

- Uma `WindowPredictionResponse` por janela, em ordem cronológica,
  com valores convertidos para tipos nativos do Python (`int`,
  `float`, `bool`).
- Sem `prediction_table`, `window_end_date` e `close_price` ficam
  `None`. Com a tabela, vêm do índice (como `date`) e da coluna
  `close_price`.
- `number_of_windows`: total de janelas. `number_of_outliers`: soma
  de `is_outlier`. `outlier_rate`: `number_of_outliers /
  number_of_windows`.

### Tradução de erros (`exception_handler`)

| Exceção levantada | HTTP | `detail` |
|---|---|---|
| `LookupError` (inclui `KeyError`) | 404 | mensagem do erro, sem as aspas que o `KeyError` adiciona |
| `ValueError` | 422 | mensagem do erro |
| `wandb.errors.CommError` com "not found" na mensagem (sem diferenciar maiúsculas) | 404 | `Model artifact not found: <erro>` |
| `wandb.errors.CommError` com qualquer outra mensagem | 503 | `wandb is unavailable: <erro>` |
| Corpo ou query inválidos (schemas da spec 06) | 422 | validação padrão do FastAPI |

## Erros esperados

Todos os erros saem como resposta HTTP, conforme a tabela acima.
Nenhuma exceção desses tipos chega ao cliente como erro 500.

## Critérios de aceite

Use `fastapi.testclient.TestClient` dentro de um `with` (para rodar o
`lifespan`). Substitua o serviço por um `MagicMock` com
`app.dependency_overrides[get_model_service]` e limpe os overrides
ao final. O mock serve um `AutoencoderPredictor` com um
`SimpleAutoencoder(input_dim=5)` sem treino, `default_ticker =
"PETR4.SA"` e `list_loaded_models` com os metadados desse preditor.

- **CA-01:** ao subir, a API chama `get_predictor()` sem argumentos.
- **CA-02:** com `AUTOENCODER_PRELOAD_MODEL=false`, a API sobe sem
  chamar `get_predictor`.
- **CA-03:** se `get_predictor` falhar na inicialização, a API sobe
  mesmo assim e `/health` responde 200.
- **CA-04:** `GET /health` retorna exatamente
  `{"status": "ok", "default_ticker": "PETR4.SA",
  "loaded_tickers": ["PETR4.SA"]}`.
- **CA-05:** `GET /models/loaded` retorna os metadados do preditor
  em memória.
- **CA-06:** `GET /models/available` retorna os modelos do serviço,
  na mesma ordem.
- **CA-07:** `GET /models/default` retorna os metadados do modelo
  padrão.
- **CA-08:** `POST /models/reload` com `{"ticker": "AAPL"}` chama
  `reload("AAPL")`, e com `{}` chama `reload(None)`.
- **CA-09:** `POST /predictions/log-returns` com 12 log-retornos
  responde 200 com 8 janelas, `window_end_index` de 4 a 11 e
  `window_end_date` igual a `null`. Com `ticker`, o serviço é chamado
  com esse ticker.
- **CA-10:** `POST /predictions/log-returns` com menos log-retornos
  que `window_size` responde 422.
- **CA-11:** `POST /predictions/prices` com 12 preços responde 200
  com 7 janelas.
- **CA-12:** `POST /predictions/prices` com um preço `0.0` responde
  422.
- **CA-13:** `GET /predictions/market` com
  `fetch_price_series` (de `predict_autoencoder`) devolvendo 10
  preços em dias úteis a partir de `2026-09-01` chama
  `fetch_price_series("PETR4.SA", "2026-09-01", "2026-09-30")` e
  responde com 5 janelas. A primeira tem `window_end_index == 4`, e a
  última tem `window_end_date == "2026-09-14"` e o `close_price` do
  último preço.
- **CA-14:** `GET /predictions/market` com `start_date` posterior a
  `end_date` responde 422.
- **CA-15:** um `CommError` com "not found" em `get_predictor`
  responde 404, e um `CommError("timeout")` em
  `list_available_models` responde 503.
- **CA-16:** um `LookupError("No servable")` em `get_predictor`
  responde 404 com `{"detail": "No servable"}`.
- **CA-17:** o `/openapi.json` contém exatamente as 8 rotas da
  tabela, e toda operação tem `summary`.

## Fora de escopo

- Autenticação e controle de acesso.
- Sinais de compra e venda.
- Imagem Docker e deploy (pasta `docker/`).
