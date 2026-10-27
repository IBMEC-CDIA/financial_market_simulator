# 02 - Normalização

- **Arquivo:** `src/generative_models/common/normalization.py`
- **Arquivo de teste:** `tests/generative_models/common/test_normalization.py`
- **Depende de:** nenhuma
- **Situação:** fornecido. O módulo e os testes já existem no repositório;
  use esta spec e os testes como referência.

## Objetivo

Padronizar (z-score) séries de log-retornos de forma que a estatística
calculada no conjunto de treino possa ser reaplicada à validação, à
série completa e, mais tarde, a dados novos na inferência, sem
vazamento de dados.

## Origem no notebook

- Célula 18: chamadas a `normalize_series` para o treino (calculando a
  média e o desvio padrão) e para a validação e a série completa
  (reutilizando a estatística do treino).

## Interface pública

```python
def normalize_series(
    series_values: torch.Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> tuple[torch.Tensor, float, float]:
    ...
```

## Comportamento

- Quando `mean` é `None`, usa a média de `series_values`.
- Quando `std` é `None`, usa o desvio padrão de `series_values`
  calculado por `torch.Tensor.std()` (estimador amostral, com
  correção de Bessel).
- Quando `mean` e/ou `std` são informados, usa exatamente esses
  valores, sem recalcular a estatística da série.
- Retorna a tupla `(serie_normalizada, media_usada, desvio_usado)`:
  - `serie_normalizada = (series_values - mean) / std`, com o mesmo
    shape da entrada;
  - `media_usada` e `desvio_usado` são `float` do Python (não
    tensores).
- Não altera o tensor de entrada.

## Erros esperados

Nenhum erro próprio. Uma série constante (desvio padrão zero) gera
valores `inf`/`nan`, e esse caso não é tratado.

## Critérios de aceite

- **CA-01:** sem `mean` e `std`, a série normalizada tem média
  aproximadamente `0` e desvio padrão aproximadamente `1` (tolerância
  de `1e-5`).
- **CA-02:** sem `mean` e `std`, a média e o desvio retornados são
  iguais a `series_values.mean().item()` e
  `series_values.std().item()`.
- **CA-03:** com `mean=2.0` e `std=4.0`, a série `[2.0, 6.0, 10.0]` é
  normalizada para `[0.0, 1.0, 2.0]`.
- **CA-04:** com `mean` e `std` informados, a função devolve
  exatamente os valores recebidos, e não a estatística da série.
- **CA-05:** a média e o desvio retornados são instâncias de `float`.
- **CA-06:** o shape da série normalizada é igual ao shape da
  entrada.

## Fora de escopo

- Outros tipos de normalização (min-max, robusta).
- Desnormalização (operação inversa).
