# 05 - Configuração do treino

- **Arquivo:** `src/generative_models/models/autoencoder/config.py`
- **Arquivo de teste:** `tests/generative_models/models/autoencoder/test_config.py`
- **Depende de:** nenhuma
- **Situação:** a implementar

## Objetivo

Substituir as constantes soltas do notebook por objetos de
configuração tipados e validados, agrupados por responsabilidade, que
possam ser criados pelo script de linha de comando (spec 08) e
enviados ao wandb como um dicionário plano.

## Origem no notebook

- Células 12, 20, 24 e 28: `START_DATE`, `END_DATE`, `WINDOW_SIZE`,
  `BATCH_SIZE`, `CONTAMINATION`, `hidden_layer_size`,
  `latent_layer_size`, `number_of_epochs`, `learning_rate`,
  `RANDOM_SEED` e `project_name`.
- Célula 18: a proporção `0.8` da separação treino/validação.

O ticker **não** faz parte da configuração: ele é passado direto ao
treinador (spec 07), para que a mesma configuração sirva para vários
tickers.

## Interface pública

Todas as classes são `dataclasses.dataclass`, com os campos e valores
padrão abaixo, nesta ordem:

```python
@dataclass
class MarketDataConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2026-09-15"
    train_split_ratio: float = 0.8
    window_size: int = 10
    batch_size: int = 32


@dataclass
class AutoencoderArchitectureConfig:
    hidden_layer_size: int = 16
    latent_layer_size: int = 4


@dataclass
class OptimizationConfig:
    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    random_seed: int = 42


@dataclass
class TrackingConfig:
    project_name: str = "autoencoder-simple-outlier-trading-agent"
    output_directory: str = "artifacts"
    env_file_path: str = ".env"


@dataclass
class AutoencoderTrainingConfig:
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    architecture: AutoencoderArchitectureConfig = field(
        default_factory=AutoencoderArchitectureConfig
    )
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    contamination: float = 0.05
    device: str | None = None

    def to_flat_dict(self) -> dict[str, object]:
        ...
```

O pacote `generative_models.models.autoencoder` deve reexportar, no
seu `__init__.py`, as cinco classes desta spec.

## Comportamento

- `AutoencoderTrainingConfig` valida os valores em `__post_init__`,
  ou seja, no momento da criação (veja **Erros esperados**). As
  sub-configurações não se validam sozinhas.
- `device=None` significa "decidir automaticamente" (a decisão é do
  treinador, spec 07).
- `to_flat_dict()` retorna um único dicionário com todos os campos
  das quatro sub-configurações, mais `contamination` e `device`,
  usando os nomes dos campos como chaves: `start_date`, `end_date`,
  `train_split_ratio`, `window_size`, `batch_size`,
  `hidden_layer_size`, `latent_layer_size`, `number_of_epochs`,
  `learning_rate`, `random_seed`, `project_name`,
  `output_directory`, `env_file_path`, `contamination` e `device`.

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| `train_split_ratio` fora do intervalo aberto `(0, 1)` | `ValueError` | `train_split_ratio must be between 0 and 1.` |
| `contamination` fora do intervalo aberto `(0, 1)` | `ValueError` | `contamination must be between 0 and 1.` |
| `window_size`, `batch_size`, `hidden_layer_size`, `latent_layer_size`, `number_of_epochs` ou `learning_rate` menor ou igual a `0` | `ValueError` | `<nome_do_campo> must be positive.` |

## Critérios de aceite

- **CA-01:** `AutoencoderTrainingConfig()` é criado sem erro e usa
  todos os valores padrão da tabela de interface.
- **CA-02:** duas instâncias padrão de `AutoencoderTrainingConfig`
  não compartilham o mesmo objeto `MarketDataConfig` (uso de
  `default_factory`).
- **CA-03:** `to_flat_dict()` contém exatamente as 15 chaves listadas
  em **Comportamento**, com os valores das sub-configurações (por
  exemplo, `MarketDataConfig(window_size=20)` resulta em
  `window_size == 20`).
- **CA-04:** `train_split_ratio` igual a `0.0`, `1.0` ou `1.5` gera
  `ValueError` (teste parametrizado).
- **CA-05:** `contamination` igual a `0.0` ou `1.0` gera
  `ValueError`.
- **CA-06:** `hidden_layer_size=0` gera `ValueError` com a mensagem
  contendo `hidden_layer_size`.
- **CA-07:** `learning_rate=-0.1` gera `ValueError` com a mensagem
  contendo `learning_rate`.

## Fora de escopo

- Leitura da configuração a partir de arquivos (YAML, JSON, TOML).
- Validação do formato das datas e do valor de `device`.
