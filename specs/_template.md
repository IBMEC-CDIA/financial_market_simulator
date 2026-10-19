# NN - Nome do módulo

Salve a spec na pasta da trilha (`training/`, `serving/` ou uma nova)
e adicione-a à tabela de ordem do `README.md` dessa trilha.

- **Arquivo:** `src/generative_models/<pacote>/<modulo>.py`
- **Arquivo de teste:** `tests/generative_models/<pacote>/test_<modulo>.py`
- **Depende de:** specs NN, NN, `<trilha>/NN` (ou "nenhuma")

## Objetivo

Duas ou três linhas dizendo qual problema o módulo resolve e quem o
utiliza.

## Origem

Células do notebook de partida ou módulos já implementados que deram
origem a este módulo, e o que muda em relação a eles.

## Interface pública

Assinaturas exatas (nomes, parâmetros, type hints e valores padrão)
de tudo o que outros módulos ou os testes utilizam.

```python
def public_function(
    first_parameter: int,
    second_parameter: str = "default",
) -> float:
    ...
```

## Comportamento

O que cada elemento da interface faz, incluindo shapes de tensores,
tipos de retorno e efeitos colaterais (arquivos gravados, chamadas
externas).

## Erros esperados

| Situação | Exceção | Mensagem (trecho) |
|---|---|---|
| ... | `ValueError` | ... |

## Critérios de aceite

Cada critério deve virar ao menos um teste.

- **CA-01:** ...
- **CA-02:** ...

## Fora de escopo

O que este módulo deliberadamente **não** faz.
