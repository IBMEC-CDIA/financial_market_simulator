# 07 - Pipeline de treino

- **Arquivo:** `src/generative_models/models/autoencoder/trainer.py`
- **Arquivo de teste:** `tests/generative_models/models/autoencoder/test_trainer.py`
- **Depende de:** specs 01 a 06
- **Situação:** a implementar

## Objetivo

Reunir em uma classe todo o treinamento que o notebook faz célula a
célula: coleta de dados, preparação, treino, validação, cálculo do
limiar de anomalia e registro do modelo, para um ticker por vez.

## Origem no notebook

| Células | Vira |
|---|---|
| 12, 16, 18, 20 | `AutoencoderTrainer.prepare_data` |
| 24 | `AutoencoderTrainer.build_model` |
| 26 | `compute_reconstruction_scores` |
| 28 | `AutoencoderTrainer._build_experiment_tracker` |
| 30 (laço de épocas) | `AutoencoderTrainer._fit` |
| 30 (validação) | `AutoencoderTrainer._evaluate` |
| 32 | `AutoencoderTrainer._compute_anomaly_threshold` |
| 30 (`log_model`, `finish_run`) | `AutoencoderTrainer.train` |

Diferenças em relação ao notebook:

- O nome do artifact e do arquivo de pesos passam a depender do
  ticker.
- Os metadados do modelo passam a incluir o limiar de anomalia e a
  estatística de normalização, que a inferência vai precisar.
- O run do wandb é encerrado mesmo se o treino falhar.

## Interface pública

```python
MODEL_NAME_PREFIX = "simple-autoencoder-"


@dataclass
class PreparedAutoencoderData:
    train_dataset: SlidingWindowReconstructionDataset
    validation_dataset: SlidingWindowReconstructionDataset
    full_dataset: SlidingWindowReconstructionDataset
    return_train_mean: float
    return_train_std: float


@dataclass
class AutoencoderTrainingResult:
    model: SimpleAutoencoder
    training_loss_history: list[float]
    validation_loss: float
    anomaly_threshold: float
    return_train_mean: float
    return_train_std: float
    model_file_path: Path


def compute_reconstruction_scores(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    ...


def build_ticker_slug(ticker: str) -> str:
    ...


class AutoencoderTrainer:
    def __init__(
        self,
        ticker: str,
        config: AutoencoderTrainingConfig,
        experiment_tracker: WandbExperimentTracker | None = None,
    ) -> None:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def model_file_path(self) -> Path:
        ...

    def prepare_data(self) -> PreparedAutoencoderData:
        ...

    def build_model(self) -> SimpleAutoencoder:
        ...

    def train(self) -> AutoencoderTrainingResult:
        ...
```

Atributos públicos de `AutoencoderTrainer`: `ticker`, `config`,
`device` e `experiment_tracker`. O pacote
`generative_models.models.autoencoder` deve reexportar
`SimpleAutoencoder`, `AutoencoderTrainer` e
`AutoencoderTrainingResult`.

`fetch_price_series` deve ser importada no topo de `trainer.py`
(`from generative_models.data.market_data import fetch_price_series`),
para que os testes possam substituí-la com
`patch("generative_models.models.autoencoder.trainer.fetch_price_series")`.

## Comportamento

### Funções auxiliares

- `compute_reconstruction_scores(model, input_tensor)`: coloca o
  modelo em `eval()`, calcula sem gradiente o erro quadrático de cada
  posição e retorna, como `np.ndarray` 1D, a **média por janela**
  (shape `(num_windows,)`).
- `build_ticker_slug(ticker)`: converte para minúsculas e troca todo
  caractere que não seja letra, dígito, `.`, `_` ou `-` por `-`.

### Construtor e propriedades

- `device` é `config.device` quando definido; caso contrário,
  `"cuda"` se disponível e `"cpu"` se não.
- Se `experiment_tracker` for `None`, cria um
  `WandbExperimentTracker` com `project_name` e `env_file_path` de
  `config.tracking` e com o `config` do run igual a
  `{"ticker": ticker, **config.to_flat_dict(), "optimizer": "Adam",
  "loss_function": "MSE", "model": "SimpleAutoencoder"}`.
- `model_name` é `MODEL_NAME_PREFIX + build_ticker_slug(ticker)`.
- `model_file_path` é
  `Path(config.tracking.output_directory) / f"{model_name}.pt"`.

### `prepare_data`

1. Busca os preços com
   `fetch_price_series(ticker, start_date, end_date)`.
2. Converte `close_price` em tensor `float32` 1D e calcula os
   log-retornos `log(p[t + 1]) - log(p[t])` com operações de tensor.
3. Separa cronologicamente (sem embaralhar): os primeiros
   `int(len(log_returns) * train_split_ratio)` valores são treino e o
   restante é validação.
4. Se treino ou validação tiver menos que `window_size` valores,
   lança `ValueError`.
5. Normaliza o treino com `normalize_series` e reaplica a média e o
   desvio do treino à validação e à série completa.
6. Retorna os três `SlidingWindowReconstructionDataset` (treino,
   validação e série completa) e a estatística do treino.

### `build_model`

Retorna um `SimpleAutoencoder` com `input_dim=window_size`,
`hidden_dim=hidden_layer_size` e `latent_dim=latent_layer_size`, já no
`device`.

### `train`

1. Chama `prepare_data`.
2. Fixa a semente com `torch.manual_seed(random_seed)` e chama
   `build_model`.
3. Chama `experiment_tracker.start_run()`.
4. Treina (`_fit`): `DataLoader` com `batch_size`, `shuffle=True` e um
   `torch.Generator` com a mesma semente; otimizador Adam com
   `learning_rate`; perda `nn.MSELoss`; `number_of_epochs` épocas.
   A cada época:
   - calcula a perda média da época, ponderada pelo tamanho de cada
     lote;
   - guarda essa perda no histórico;
   - chama `log_metrics({"epoch": <época começando em 1>,
     "train/loss_mse": <perda média>})`.
5. Avalia (`_evaluate`) a MSE média nas janelas de validação, sem
   embaralhar e sem gradiente, e chama
   `log_metrics({"validation/loss_mse": <perda>})`.
6. Calcula o limiar (`_compute_anomaly_threshold`): o percentil
   `100 * (1 - contamination)` dos escores de
   `compute_reconstruction_scores` sobre **todas** as janelas da
   série completa, como `float`.
7. Cria o diretório de `model_file_path`, se não existir, e chama
   `log_model` com `model_name`, `str(model_file_path)`, uma
   descrição citando o ticker e
   `metadata={"final_train_loss", "validation_loss",
   "anomaly_threshold", "return_train_mean", "return_train_std"}`.
8. Chama `experiment_tracker.finish_run()` **sempre**, mesmo quando um
   dos passos 4 a 7 lança exceção (`try`/`finally`). A exceção
   original continua sendo propagada.
9. Retorna um `AutoencoderTrainingResult` com o modelo treinado, o
   histórico, a perda de validação, o limiar, a estatística de
   normalização e `model_file_path`.

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| Treino ou validação com menos de `window_size` log-retornos | `ValueError` | `Not enough data for ticker '<ticker>'` |

## Critérios de aceite

Nos testes, substitua `fetch_price_series` por um mock que retorne
uma série sintética positiva (por exemplo, 101 preços gerados por
`100 * exp(cumsum(normal(0, 0.01)))`), passe um `MagicMock` como
`experiment_tracker` e use uma configuração pequena:
`window_size=5`, `batch_size=8`, `number_of_epochs=3`,
`device="cpu"` e `output_directory` dentro de `tmp_path`.

- **CA-01:** `build_ticker_slug("PETR4.SA") == "petr4.sa"` e
  `build_ticker_slug("^BVSP") == "-bvsp"`.
- **CA-02:** `compute_reconstruction_scores` com 7 janelas de tamanho
  5 retorna um `np.ndarray` de shape `(7,)` com valores `>= 0`.
- **CA-03:** para o ticker `"PETR4.SA"`, `model_name` é
  `"simple-autoencoder-petr4.sa"` e `model_file_path` é
  `<output_directory>/simple-autoencoder-petr4.sa.pt`.
- **CA-04:** com 101 preços (100 log-retornos) e
  `train_split_ratio=0.8`, os datasets têm 76 (treino), 16
  (validação) e 96 (série completa) janelas.
- **CA-05:** a série armazenada no dataset de treino tem média
  aproximadamente `0`, e `return_train_std > 0`.
- **CA-06:** com 12 preços e `window_size=5`, `prepare_data` lança
  `ValueError` com `Not enough data`.
- **CA-07:** `train` chama `start_run` uma vez, `log_model` uma vez e
  `finish_run` uma vez.
- **CA-08:** com `number_of_epochs=3`, `log_metrics` recebe as chaves
  `epoch` e `train/loss_mse` três vezes (épocas 1, 2 e 3) e
  `validation/loss_mse` uma vez.
- **CA-09:** o resultado tem `len(training_loss_history) == 3`, um
  `validation_loss` finito e um `anomaly_threshold` `float` maior que
  `0`.
- **CA-10:** os `metadata` passados a `log_model` contêm as cinco
  chaves do passo 7 do `train`.
- **CA-11:** se `log_model` lançar uma exceção, `train` propaga a
  exceção e `finish_run` é chamado mesmo assim.
- **CA-12:** sem `experiment_tracker`, o tracker padrão é criado com
  `project_name` de `config.tracking` e com um `config` que contém
  `ticker`, todas as chaves de `to_flat_dict()`, `optimizer`,
  `loss_function` e `model` (substitua `WandbExperimentTracker` por
  um mock no módulo `trainer`).
- **CA-13:** com a mesma configuração e a mesma série, duas execuções
  de `train` produzem o mesmo `training_loss_history`
  (reprodutibilidade).

## Fora de escopo

- Treinar vários tickers de uma vez.
- Early stopping, agendador de taxa de aprendizado ou checkpoints
  intermediários.
- Alinhamento das janelas outlier com as datas da série, sinais de
  compra e venda e gráficos (células 33 em diante do notebook).
