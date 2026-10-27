# 04 - Autoencoder denso

- **Arquivo:** `src/generative_models/models/autoencoder/autoencoder.py`
- **Arquivo de teste:** `tests/generative_models/models/autoencoder/test_autoencoder.py`
- **Depende de:** nenhuma
- **Situação:** a implementar

## Objetivo

Definir a arquitetura do autoencoder totalmente conectado que
reconstrói janelas de log-retornos normalizados. O erro de
reconstrução é usado como escore de anomalia.

## Origem no notebook

- Célula 22: classe `SimpleAutoencoder`. A única diferença é o novo
  método `encode`, que expõe a representação latente.
- Célula 24: instanciação com `input_dim=WINDOW_SIZE`.

## Interface pública

```python
class SimpleAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        ...

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
```

Atributos públicos: `encoder` e `decoder`, ambos `nn.Sequential`.

## Comportamento

A arquitetura deve ser **exatamente** esta, porque os pesos salvos
(`state_dict`) precisam ser compatíveis entre o treino e a inferência:

| Bloco | Camadas, na ordem |
|---|---|
| `encoder` | `Linear(input_dim, hidden_dim)`, `ReLU`, `Linear(hidden_dim, latent_dim)`, `ReLU` |
| `decoder` | `Linear(latent_dim, hidden_dim)`, `ReLU`, `Linear(hidden_dim, input_dim)` |

- `encode(x)` aplica apenas o `encoder` e retorna um tensor de shape
  `(batch_size, latent_dim)`.
- `forward(x)` aplica `encode` seguido do `decoder` e retorna um
  tensor de shape `(batch_size, input_dim)`.
- O `decoder` não tem ativação final: a saída pode ser negativa,
  assim como os log-retornos normalizados.
- Como o `encoder` termina em `ReLU`, a representação latente nunca
  é negativa.

## Erros esperados

Nenhum erro próprio. Entradas com a última dimensão diferente de
`input_dim` geram o erro padrão do PyTorch.

## Critérios de aceite

- **CA-01:** com `input_dim=10`, uma entrada `(8, 10)` produz uma
  saída `(8, 10)` em `forward`.
- **CA-02:** com `latent_dim=3`, `encode` de uma entrada `(8, 10)`
  produz `(8, 3)`.
- **CA-03:** os valores retornados por `encode` são todos `>= 0`.
- **CA-04:** em modo `eval()`, duas chamadas de `forward` com a mesma
  entrada retornam tensores idênticos.
- **CA-05:** sem argumentos opcionais, `hidden_dim=16` e
  `latent_dim=4` (verifique `encoder[0].out_features` e
  `encoder[2].out_features`).
- **CA-06:** as chaves do `state_dict` são exatamente
  `encoder.0.weight`, `encoder.0.bias`, `encoder.2.weight`,
  `encoder.2.bias`, `decoder.0.weight`, `decoder.0.bias`,
  `decoder.2.weight` e `decoder.2.bias`.

## Fora de escopo

- Camadas recorrentes (LSTM), dropout ou normalização em lote.
- Profundidade configurável.
