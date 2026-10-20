# 05 - Serviço de modelos (singleton)

- **Arquivo:** `src/generative_models/serving/model_service.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_model_service.py`
- **Depende de:** specs 03 e 04
- **Situação:** a implementar

## Objetivo

Garantir que, dentro de um processo (a API), cada modelo seja baixado
do wandb **uma única vez** e reutilizado por todas as requisições.
O serviço também define o "modelo padrão", que é usado quando a
requisição não informa um ticker.

## Origem

- Spec 04: o carregamento reutiliza `load_predictor`, a mesma função
  do script de predição.
- Spec 03: o cache em memória é o próprio `models` do
  `WandbAutoencoderRegistry`.

## Interface pública

```python
@dataclass
class AvailableModel:
    artifact_name: str
    ticker: str
    created_at: str


class AutoencoderModelService:
    def __init__(self, registry: WandbAutoencoderRegistry) -> None:
        ...

    @classmethod
    def get_instance(cls) -> "AutoencoderModelService":
        ...

    @classmethod
    def reset_instance(cls) -> None:
        ...

    def get_predictor(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        ...

    def reload(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        ...

    def list_loaded_models(self) -> list[AutoencoderModelMetadata]:
        ...

    def list_available_models(self) -> list[AvailableModel]:
        ...
```

Atributos públicos: `registry` e `default_ticker` (`None` até o modelo
padrão ser carregado).

`load_predictor` e `WandbAutoencoderRegistry` devem ser importados no
topo do módulo, para que os testes possam substituí-los com
`patch("generative_models.serving.model_service.<nome>")`.

## Comportamento

### Singleton

- `get_instance()` cria a instância na primeira chamada e devolve a
  mesma instância nas seguintes.
- A criação é **thread-safe**: várias threads chamando
  `get_instance()` ao mesmo tempo criam um único registro (use um
  `threading.Lock` com verificação dupla).
- O registro é criado com as variáveis de ambiente:

  | Variável | Parâmetro do registro | Padrão |
  |---|---|---|
  | `AUTOENCODER_WANDB_PROJECT` | `project_name` | `TrackingConfig.project_name` |
  | `AUTOENCODER_WANDB_ENTITY` | `entity` | `None` |
  | `AUTOENCODER_DEVICE` | `device` | `None` |

- `reset_instance()` descarta a instância. A próxima chamada a
  `get_instance()` cria uma nova (usado nos testes).

### `get_predictor(ticker=None)`

- Sem ticker:
  - se ainda não há modelo padrão, chama `reload()` (que define o
    padrão);
  - se já há, retorna `registry.get_model(default_ticker)` sem
    acessar o wandb.
- Com ticker: converte para maiúsculas. Se o ticker já está em
  `registry.models`, retorna o modelo em memória; caso contrário,
  carrega com `load_predictor(registry, ticker_maiusculo)`.

### `reload(ticker=None)`

- Sempre baixa de novo com `load_predictor`, que substitui o modelo
  em memória.
- Sem ticker: carrega o modelo mais recente de qualquer ticker e o
  torna o novo `default_ticker`.
- Com ticker: recarrega o modelo desse ticker (em maiúsculas) **sem**
  mudar o `default_ticker`.

### Listagens

- `list_loaded_models()`: metadados dos preditores em
  `registry.models`, na ordem de carregamento.
- `list_available_models()`: um `AvailableModel` por artifact de
  `registry.list_available_artifacts()`, com `artifact_name` no
  formato `"<colecao>:<versao>"`, o `ticker` dos metadados e
  `created_at` como `str`, **ordenados do mais recente para o mais
  antigo**. Nada é baixado.

### Concorrência

`get_predictor`, `reload` e `list_loaded_models` usam um mesmo
`threading.RLock`, para que duas requisições simultâneas não baixem o
mesmo modelo duas vezes.

## Erros esperados

Os erros do registro (spec 03) são propagados sem tratamento. A
tradução para respostas HTTP é feita na API (spec 07).

## Critérios de aceite

Chame `AutoencoderModelService.reset_instance()` antes e depois de
cada teste (fixture `autouse`). Substitua `WandbAutoencoderRegistry` e
`load_predictor` por mocks.

- **CA-01:** duas chamadas a `get_instance()` retornam o mesmo
  objeto, e o registro é criado uma única vez.
- **CA-02:** 10 threads chamando `get_instance()` ao mesmo tempo
  recebem a mesma instância, e o registro é criado uma única vez.
- **CA-03:** com as três variáveis de ambiente definidas, o registro
  é criado com `project_name`, `entity` e `device` iguais a elas.
- **CA-04:** após `reset_instance()`, `get_instance()` cria uma nova
  instância.
- **CA-05:** duas chamadas a `get_predictor()` retornam o mesmo
  preditor, `load_predictor` é chamado uma vez com `(registry, None)`,
  e `default_ticker` passa a ser o ticker do preditor.
- **CA-06:** `get_predictor("petr4.sa")` com `PETR4.SA` já em memória
  não chama `load_predictor`.
- **CA-07:** `get_predictor("aapl")` sem o modelo em memória chama
  `load_predictor(registry, "AAPL")`.
- **CA-08:** `reload()` define `default_ticker` com o ticker do
  preditor carregado.
- **CA-09:** `reload("aapl")` chama `load_predictor(registry,
  "AAPL")` e mantém o `default_ticker` anterior.
- **CA-10:** `list_available_models` com artifacts criados em datas
  diferentes retorna a lista ordenada do mais recente para o mais
  antigo, com `artifact_name` versionado.

## Fora de escopo

- Expirar ou descarregar modelos da memória (exceto por
  `reset_instance`).
- Recarregar automaticamente quando um novo artifact é registrado.
