# 03 - Dataset de janelas deslizantes

- **Arquivo:** `src/generative_models/data/datasets.py`
- **Arquivo de teste:** `tests/generative_models/data/test_datasets.py`
- **Depende de:** nenhuma

## Objetivo

Transformar uma série temporal 1D em amostras de janelas deslizantes
para modelos de reconstrução (autoencoders e VAEs), em que a janela é,
ao mesmo tempo, a entrada e o alvo do modelo.

## Origem no notebook

- Célula 20: criação de `SlidingWindowReconstructionDataset` para
  treino, validação e série completa, consumidos por `DataLoader`.

## Interface pública

```python
class SlidingWindowReconstructionDataset(Dataset):
    def __init__(
        self,
        series_values: np.ndarray | torch.Tensor,
        window_size: int,
    ) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> torch.Tensor:
        ...
```

Atributos públicos, usados pelos testes da spec 07:

- `series_values: torch.Tensor`: a série armazenada.
- `window_size: int`: o tamanho da janela.

## Comportamento

- Herda de `torch.utils.data.Dataset`.
- No construtor:
  - um `torch.Tensor` é armazenado como está;
  - um `np.ndarray` é convertido com `torch.from_numpy` (o `dtype`
    original é preservado);
  - qualquer outro tipo gera `TypeError`.
- `len(dataset)` é `len(series_values) - window_size + 1`.
- `dataset[i]` retorna o tensor 1D
  `series_values[i : i + window_size]`, com shape `(window_size,)`.
  Não existe janela de saída separada.
- As janelas são consecutivas com passo 1 e se sobrepõem.
- Funciona com `DataLoader`: um lote de `B` janelas tem shape
  `(B, window_size)`.

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| `series_values` não é `np.ndarray` nem `torch.Tensor` (por exemplo, uma `list`) | `TypeError` | `series_values must be a numpy.ndarray or a torch.Tensor` |

## Critérios de aceite

- **CA-01:** para uma série com 10 valores e `window_size=3`,
  `len(dataset) == 8`.
- **CA-02:** a partir de um `np.ndarray` `[0, 1, ..., 9]` e
  `window_size=3`, `dataset[2]` é o tensor `[2, 3, 4]`.
- **CA-03:** o mesmo resultado do CA-02 vale quando a série é um
  `torch.Tensor`.
- **CA-04:** todas as janelas têm shape `(window_size,)`, inclusive a
  última (`dataset[len(dataset) - 1]`).
- **CA-05:** passar uma `list` em `series_values` gera `TypeError`.
- **CA-06:** com `DataLoader(dataset, batch_size=4)`, o primeiro lote
  tem shape `(4, window_size)`.

## Fora de escopo

- Validação de `window_size` maior que a série ou não positivo.
- Passo (stride) diferente de 1.
- Janelas de entrada e saída separadas (previsão).
