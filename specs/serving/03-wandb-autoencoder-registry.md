# 03 - Registro de modelos no wandb

- **Arquivo:** `src/generative_models/serving/wandb_autoencoder_registry.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_wandb_autoencoder_registry.py`
- **Depende de:** specs 01 e 02, `training/05`, `training/07`

## Objetivo

Implementar o `ModelRegistry` (spec 02) para os artifacts que o
`AutoencoderTrainer` registra no wandb: listar os modelos
disponíveis, baixar os pesos, recriar o `SimpleAutoencoder` com a
arquitetura salva nos metadados e devolver um `AutoencoderPredictor`
(spec 01) pronto para uso.

## Origem

- `training/07`: nome do artifact (`MODEL_NAME_PREFIX` +
  `build_ticker_slug(ticker)`) e metadados gravados por `log_model`.
- `training/04`: a arquitetura precisa ser recriada com os mesmos
  tamanhos antes de `load_state_dict`.
- `training/05`: o projeto padrão é `TrackingConfig.project_name`.

## Interface pública

```python
REQUIRED_METADATA_KEYS = (
    "ticker",
    "window_size",
    "hidden_layer_size",
    "latent_layer_size",
    "return_train_mean",
    "return_train_std",
    "anomaly_threshold",
)


class WandbAutoencoderRegistry(ModelRegistry):
    def __init__(
        self,
        project_name: str = TrackingConfig.project_name,
        entity: str | None = None,
        download_directory: str = "artifacts/wandb",
        env_file_path: str = ".env",
        device: str | None = None,
    ) -> None:
        ...

    @property
    def api(self) -> wandb.Api:
        ...

    @property
    def project_path(self) -> str:
        ...

    def list_available_artifacts(self) -> list[wandb.Artifact]:
        ...

    def load_latest_model(self) -> AutoencoderPredictor:
        ...

    def load_model(self, ticker: str) -> AutoencoderPredictor:
        ...

    def load_all_models(self) -> list[str]:
        ...

    def get_model(self, ticker: str) -> AutoencoderPredictor:
        ...

    def list_loaded_tickers(self) -> list[str]:
        ...
```

Atributos públicos: `models` (herdado), `project_name`, `entity`,
`download_directory` (`Path`), `env_file_path` e `device`.

O módulo deve importar `wandb` no topo (`import wandb`), para que os
testes possam substituir o cliente com
`patch("generative_models.serving.wandb_autoencoder_registry.wandb.Api")`.

## Comportamento

### Construtor e propriedades

- `device` é o valor recebido ou, se `None`, `"cuda"` quando
  disponível e `"cpu"` quando não.
- O construtor **não** acessa o wandb.
- `api`: na primeira leitura, carrega o `.env` de `env_file_path`
  (`load_dotenv`) e cria `wandb.Api()`. As leituras seguintes
  reutilizam o mesmo cliente.
- `project_path`: `"<entity>/<project_name>"`, em que `entity` é o
  valor recebido ou, se `None`, `api.default_entity`.

### `list_available_artifacts`

- Percorre `api.artifact_collections(project_name=project_path,
  type_name="model")`.
- Ignora coleções cujo nome não começa com `MODEL_NAME_PREFIX`.
- Para as demais, busca a versão `latest`
  (`api.artifact("<project_path>/<colecao>:latest")`) e só a mantém
  se os metadados tiverem **todas** as chaves de
  `REQUIRED_METADATA_KEYS`.

### `load_latest_model`

- Entre os artifacts de `list_available_artifacts`, carrega o de
  maior `created_at`.

### `load_model(ticker)`

- Busca `"<project_path>/simple-autoencoder-<slug>:latest"`, com
  `type="model"`, em que `slug` é `build_ticker_slug(ticker)`.
- Se faltar alguma chave obrigatória nos metadados, lança
  `ValueError`. Caso contrário, carrega o artifact.

### Carregamento de um artifact (método interno)

1. Monta o nome versionado `"<colecao>:<versao>"` (por exemplo,
   `"simple-autoencoder-aapl:v3"`), a partir do nome do artifact sem
   o sufixo após `:` e de `artifact.version`.
2. Baixa o artifact em
   `download_directory / "<colecao>-<versao>"` (o `:` vira `-`).
3. Usa o primeiro arquivo `.pt` do diretório (em ordem alfabética).
   Se não houver nenhum, lança `FileNotFoundError`.
4. Recria o `SimpleAutoencoder` com `input_dim=window_size`,
   `hidden_dim=hidden_layer_size` e `latent_dim=latent_layer_size`
   (todos convertidos para `int`) e carrega os pesos com
   `torch.load(..., map_location=device, weights_only=True)`.
5. Cria o `AutoencoderPredictor` com o modelo no `device` e um
   `AutoencoderModelMetadata` montado a partir dos metadados
   (`ticker` como `str`, `window_size` como `int`, estatísticas e
   limiar como `float`) e do nome versionado.
6. Guarda o preditor em `models[metadata.ticker]`, substituindo um
   preditor anterior do mesmo ticker, e o retorna.

### `load_all_models`, `get_model`, `list_loaded_tickers`

- `load_all_models()`: carrega todos os artifacts de
  `list_available_artifacts` e retorna seus tickers.
- `get_model(ticker)`: retorna `models[ticker]` ou lança `KeyError`
  se o ticker não foi carregado. Nunca acessa o wandb.
- `list_loaded_tickers()`: tickers de `models`, na ordem de
  carregamento.

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| Nenhum artifact servível no projeto (`load_latest_model`) | `LookupError` | `No servable autoencoder artifact found in '<project_path>'.` |
| Artifact do ticker sem metadados obrigatórios (`load_model`) | `ValueError` | `is missing metadata required for inference` |
| Artifact sem arquivo `.pt` | `FileNotFoundError` | `has no .pt weights file.` |
| `get_model` de ticker não carregado | `KeyError` | `No model loaded for ticker '<ticker>'. Call load_model first.` |
| Artifact inexistente ou wandb fora do ar | `wandb.errors.CommError` | propagado sem tratamento |

## Critérios de aceite

Substitua `wandb.Api` por um mock. Monte artifacts falsos
(`MagicMock`) com `name`, `version`, `created_at`, `metadata` e um
`download(root)` que grava em `root` o `state_dict` de um
`SimpleAutoencoder(input_dim=5, hidden_dim=8, latent_dim=2)`. Use
`entity="test-entity"`, `device="cpu"` e diretórios em `tmp_path`.

- **CA-01:** o construtor não chama `wandb.Api`. A primeira leitura
  de `api` cria o cliente, e a segunda reutiliza o mesmo objeto.
- **CA-02:** com `entity=None`, `project_path` usa
  `api.default_entity`. Com `entity="test-entity"` e
  `project_name="test-project"`, é `"test-entity/test-project"`.
- **CA-03:** `list_available_artifacts` ignora coleções sem o prefixo
  `simple-autoencoder-` e artifacts sem alguma chave obrigatória.
- **CA-04:** com dois artifacts válidos, `load_latest_model` carrega
  o de `created_at` mais recente.
- **CA-05:** sem artifacts válidos, `load_latest_model` lança
  `LookupError`.
- **CA-06:** `load_model("PETR4.SA")` busca
  `"test-entity/test-project/simple-autoencoder-petr4.sa:latest"` e
  devolve um `AutoencoderPredictor` com `window_size`,
  `return_train_mean`, `return_train_std` e `anomaly_threshold`
  iguais aos metadados.
- **CA-07:** o `artifact_name` do preditor é o nome versionado (por
  exemplo, `"simple-autoencoder-petr4.sa:v2"`), e não `":latest"`.
- **CA-08:** o modelo recriado tem a arquitetura dos metadados
  (`hidden_dim=8`, `latent_dim=2`) e os mesmos pesos gravados pelo
  `download` falso.
- **CA-09:** `load_model` com metadados incompletos lança
  `ValueError`.
- **CA-10:** um artifact cujo `download` não grava nenhum `.pt` gera
  `FileNotFoundError`.
- **CA-11:** `load_all_models` retorna os tickers de todos os
  artifacts válidos, e `list_loaded_tickers` passa a listá-los.
- **CA-12:** `get_model` de um ticker não carregado lança `KeyError`
  sem chamar a API.

## Fora de escopo

- Escolher uma versão específica do artifact (sempre `latest`).
- Cache entre processos: o download se repete a cada novo processo.
- Tratar erros de rede do wandb (é papel da API, spec 07).
