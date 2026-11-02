# 01 - Estrutura de pacote

- **Arquivo:** `src/generative_models/models/<nome_do_modelo>/__init__.py`
- **Arquivo de teste:** `tests/generative_models/models/<nome_do_modelo>/__init__.py`
- **Depende de:** nenhuma
- **Situação:** a implementar

## Objetivo

Definir onde cada modelo fica no repositório, para que o CI encontre os
testes e o pipeline encontre os módulos de treino e de avaliação.

## Regras

1. Cada modelo é um pacote em `src/generative_models/models/`, com um nome
   em `snake_case` que termina no tipo do modelo, por exemplo
   `lstm_autoencoder`.
2. O pacote tem um `__init__.py` com uma docstring de uma linha que diz o
   que o modelo faz.
3. O pacote tem, no mínimo, estes módulos:
   - `run_train_script.py`, com o comando de treino da spec 02;
   - `overfitting.py`, com o cálculo do relatório da spec 04.
4. Os testes espelham o pacote: `tests/generative_models/models/<nome_do_modelo>/`
   tem um `__init__.py` vazio e um `test_<modulo>.py` para cada módulo.
5. Código reutilizável por outros modelos, como a normalização, o dataset
   de janelas ou o rastreamento do wandb, fica nos pacotes `data/`,
   `tracking/` e `registry/`. Não é copiado para dentro de cada modelo.
6. Nomes de arquivos de saída seguem o prefixo `simple-autoencoder-` do
   modelo atual, ou `<nome_do_modelo>-`, e nunca usam espaços.

## Critérios de aceite

- **CA-01:** existe `src/generative_models/models/<nome>/__init__.py` com
  docstring.
- **CA-02:** existe `tests/generative_models/models/<nome>/__init__.py`.
- **CA-03:** o pacote tem `run_train_script.py` e `overfitting.py`, e o
  comando `python -m generative_models.models.<nome>.run_train_script --help`
  roda sem erro.
- **CA-04:** nenhum módulo do pacote importa de outro pacote de modelo.

## Fora de escopo

- A arquitetura interna do modelo (camadas, número de unidades).
- A escolha de hiperparâmetros.
