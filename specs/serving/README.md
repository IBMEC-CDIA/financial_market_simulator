# Trilha de serving

Disponibiliza para inferência os autoencoders treinados na
[trilha de treino](../training/README.md) e registrados no wandb.
Ao final da trilha, o mesmo modelo pode ser usado de duas formas:

```bash
uv run python -m generative_models.serving.predict_autoencoder \
    --ticker PETR4.SA

uv run uvicorn generative_models.serving.api:app --reload
```

O fluxo de trabalho, as regras gerais e a definição de pronto estão
no [README das specs](../README.md). Esta trilha só começa depois que
todas as specs de treino estiverem verdes.

## Ordem das specs

| # | Spec | Arquivo | Depende de |
|---|---|---|---|
| 01 | [Preditor do autoencoder](01-autoencoder-predictor.md) | `serving/autoencoder_predictor.py` | `training/02`, `training/04`, `training/07` |
| 02 | [Interface de registro de modelos](02-model-registry.md) | `serving/model_registry.py` | - |
| 03 | [Registro de modelos no wandb](03-wandb-autoencoder-registry.md) | `serving/wandb_autoencoder_registry.py` | 01, 02, `training/05`, `training/07` |
| 04 | [Script de predição](04-predict-cli.md) | `serving/predict_autoencoder.py` | 01, 03, `training/01` |
| 05 | [Serviço de modelos (singleton)](05-model-service.md) | `serving/model_service.py` | 03, 04 |
| 06 | [Schemas da API](06-api-schemas.md) | `serving/api_schemas.py` | - |
| 07 | [API REST](07-api.md) | `serving/api.py` | 01, 04, 05, 06 |

Os caminhos de **Arquivo** são relativos a `src/generative_models/`.
Os testes ficam em `tests/generative_models/serving/`.

## Como as peças se encaixam

```
wandb (artifacts do treino)
   |
   v
WandbAutoencoderRegistry (03) --herda--> ModelRegistry (02)
   |  baixa o .pt, recria o SimpleAutoencoder e monta
   v
AutoencoderPredictor (01)  <-- normaliza, cria janelas e pontua
   |
   +--> predict_autoencoder (04): script de terminal
   |
   +--> AutoencoderModelService (05): mantém os preditores em memória
           |
           v
        api (07) + api_schemas (06): rotas FastAPI
```

## Mapa notebook e treino -> módulos

| Origem | O que faz | Spec |
|---|---|---|
| Notebook, célula 18 | Normalizar com a média e o desvio do treino | 01 |
| Notebook, células 20 e 32 | Janelas da série completa, escore e limiar | 01 |
| Notebook, célula 35 | Alinhar cada janela ao seu índice final | 01, 04 |
| `training/07` (`log_model`) | Artifact e metadados que precisam ser lidos de volta | 03 |
| `training/04` | Arquitetura que precisa ser recriada antes de `load_state_dict` | 03 |
| `training/01` | Coleta de preços para pontuar um período recente | 04, 07 |

## Dependências extras

A API usa `fastapi`, `uvicorn` e `pydantic` (instalado com o
`fastapi`), já declarados em `pyproject.toml`. Os testes da API usam
`fastapi.testclient.TestClient`, que depende do `httpx` (extra `dev`).
