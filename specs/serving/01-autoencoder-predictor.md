# 01 - Preditor do autoencoder

- **Arquivo:** `src/generative_models/serving/autoencoder_predictor.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_autoencoder_predictor.py`
- **Depende de:** `training/02`, `training/04`, `training/07`

## Objetivo

Aplicar um autoencoder já treinado a dados novos: normalizar os
log-retornos com a estatística do treino, montar as janelas, calcular
o erro de reconstrução de cada janela e marcar como outlier as janelas
acima do limiar salvo no treino. Este módulo não sabe de onde o modelo
veio (wandb, disco): ele recebe o modelo e os metadados prontos.

## Origem

- Notebook, célula 18: a normalização reaproveita `return_train_mean`
  e `return_train_std`.
- Notebook, células 20 e 32: janelas da série inteira, escore com
  `compute_reconstruction_scores` e comparação com
  `anomaly_threshold`.
- Notebook, célula 35: `window_end_indices`, que liga cada janela ao
  índice do seu último log-retorno.

A diferença em relação ao notebook é que o limiar **não** é
recalculado: ele vem dos metadados do treino.

## Interface pública

```python
@dataclass
class AutoencoderModelMetadata:
    ticker: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float
    artifact_name: str


@dataclass
class AutoencoderPrediction:
    reconstruction_scores: np.ndarray
    is_outlier: np.ndarray
    window_end_indices: np.ndarray


class AutoencoderPredictor:
    def __init__(
        self,
        model: SimpleAutoencoder,
        metadata: AutoencoderModelMetadata,
        device: str = "cpu",
    ) -> None:
        ...

    def predict(
        self,
        log_return_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        ...

    def predict_from_prices(
        self,
        price_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        ...
```

Atributos públicos de `AutoencoderPredictor`: `model`, `metadata` e
`device`.

## Comportamento

### Construtor

- Guarda o modelo **em modo `eval()`**, junto com os metadados e o
  `device`. Não move o modelo de device: ele já deve estar em
  `device`.

### `predict`

1. Converte a entrada em um tensor `float32` 1D, como uma **cópia**
   (arrays `numpy` somente leitura e tensores com gradiente devem
   funcionar sem alterar a origem).
2. Se a série tiver menos que `window_size` valores, lança
   `ValueError`.
3. Normaliza com `normalize_series(..., mean=return_train_mean,
   std=return_train_std)`.
4. Monta **todas** as janelas deslizantes de tamanho `window_size`,
   com passo 1 (`n - window_size + 1` janelas para `n` log-retornos).
5. Calcula os escores com `compute_reconstruction_scores`
   (`training/07`), com as janelas no `device`.
6. Retorna um `AutoencoderPrediction` com:
   - `reconstruction_scores`: array 1D com um escore por janela;
   - `is_outlier`: array booleano, `True` quando o escore é
     **estritamente maior** que `anomaly_threshold`;
   - `window_end_indices`: `np.arange(window_size - 1, n)`, o índice
     do último log-retorno de cada janela.

### `predict_from_prices`

- Converte os preços em tensor `float32` 1D, calcula os log-retornos
  `log(p[t + 1]) - log(p[t])` e chama `predict` com eles.
- Com `m` preços, há `m - 1` log-retornos. O índice final `i` de uma
  janela se refere ao retorno entre os preços `i` e `i + 1`.

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| Menos log-retornos que `window_size` | `ValueError` | `At least <window_size> log-returns are required, got <n>.` |
| Menos preços que `window_size + 1` em `predict_from_prices` | `ValueError` | mesma mensagem, gerada por `predict` |

## Critérios de aceite

Use um `SimpleAutoencoder(input_dim=5)` sem treino, com
`torch.manual_seed` fixo, e metadados com `window_size=5`.

- **CA-01:** o construtor deixa o modelo em modo de avaliação
  (`predictor.model.training is False`).
- **CA-02:** com 12 log-retornos, `predict` devolve 8 escores, 8
  flags e `window_end_indices == [4, 5, ..., 11]`.
- **CA-03:** com `anomaly_threshold` igual a `0.0`, todas as janelas
  são outlier. Com um limiar muito alto (por exemplo, `1e9`), nenhuma
  é.
- **CA-04:** um escore exatamente igual ao limiar **não** é outlier.
- **CA-05:** a normalização usa os metadados: os escores de
  `predict(x)` com `mean=m, std=s` são iguais aos escores de
  `compute_reconstruction_scores` aplicados às janelas de
  `(x - m) / s` montadas manualmente.
- **CA-06:** com 4 log-retornos e `window_size=5`, `predict` lança
  `ValueError` com `At least 5 log-returns are required`.
- **CA-07:** `predict_from_prices(precos)` produz os mesmos escores
  que `predict(np.diff(np.log(precos)))`.
- **CA-08:** `predict` aceita `np.ndarray`, `torch.Tensor` e um array
  `numpy` somente leitura (`array.setflags(write=False)`), e não
  altera a entrada.

## Fora de escopo

- Carregar o modelo de qualquer origem (specs 02 e 03).
- Associar as janelas a datas (spec 04).
- Recalcular o limiar de anomalia.
