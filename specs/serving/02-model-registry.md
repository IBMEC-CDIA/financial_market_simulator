# 02 - Interface de registro de modelos

- **Arquivo:** `src/generative_models/serving/model_registry.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_model_registry.py`
- **Depende de:** nenhuma
- **Situação:** fornecido. O módulo e os testes já existem no repositório;
  use esta spec e os testes como referência.

## Objetivo

Definir a interface comum de qualquer registro que lê modelos
treinados de uma origem (wandb, disco local etc.), guarda esses
modelos em memória e devolve o modelo de um ticker. Quem usa um
registro depende só desta interface, e não da origem dos modelos.

## Origem

Não existe no notebook, onde o modelo vive em uma variável
(`dense_model`). A interface surge quando o modelo passa a ser
treinado em um processo (trilha de treino) e usado em outro (esta
trilha).

## Interface pública

```python
class ModelRegistry:
    def __init__(self) -> None:
        ...

    def load_model(self, ticker: str) -> Any:
        ...

    def load_all_models(self) -> list[str]:
        ...

    def get_model(self, ticker: str) -> Any:
        ...

    def list_loaded_tickers(self) -> list[str]:
        ...
```

Atributo público: `models: dict[str, Any]`, com os modelos em memória
indexados pelo ticker.

## Comportamento

- O construtor cria um dicionário `models` vazio, **próprio de cada
  instância**.
- Os quatro métodos apenas declaram o contrato e lançam
  `NotImplementedError`. As subclasses (spec 03) os implementam:
  - `load_model(ticker)`: lê o modelo do ticker, guarda em `models` e
    o retorna;
  - `load_all_models()`: lê todos os modelos disponíveis e retorna os
    tickers carregados;
  - `get_model(ticker)`: retorna o modelo já em memória;
  - `list_loaded_tickers()`: retorna os tickers em memória.

## Erros esperados

| Situação | Exceção |
|---|---|
| Qualquer um dos quatro métodos chamado em `ModelRegistry` | `NotImplementedError` |

## Critérios de aceite

- **CA-01:** uma instância nova tem `models == {}`.
- **CA-02:** duas instâncias não compartilham o mesmo dicionário
  `models` (alterar um não altera o outro).
- **CA-03:** `load_model("AAPL")`, `load_all_models()`,
  `get_model("AAPL")` e `list_loaded_tickers()` lançam
  `NotImplementedError` (teste parametrizado).
- **CA-04:** uma subclasse de teste que implementa os quatro métodos
  usando `self.models` funciona sem chamar a implementação da classe
  base, e `isinstance(subclasse(), ModelRegistry)` é `True`.

## Fora de escopo

- Qualquer origem concreta de modelos (spec 03).
- Uso de `abc.ABC`: a classe base pode ser instanciada, e a
  verificação acontece na chamada dos métodos.
