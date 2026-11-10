# 05 - Portões de qualidade e critérios de pronto

- **Arquivo:** `.github/workflows/ci.yml` (CI) e `.github/workflows/train.yml` (treino)
- **Arquivo de teste:** o próprio CI, que roda em cada push e pull request
- **Depende de:** specs 01 a 04
- **Situação:** fornecido pelo projeto

## Objetivo

Listar as checagens que um modelo novo precisa passar, e a definição de
pronto para ele entrar no `main`.

## Checagens do CI

| Job | O que verifica | Falha quando |
|---|---|---|
| `structure` | arquivos obrigatórios, pasta `tests/`, `.env` ignorado, `uv.lock` sincronizado | falta arquivo, ou o lock está desatualizado |
| `lint` | pylint em `src/generative_models` e `tests` | qualquer aviso do pylint |
| `unit-test` | pytest com cobertura de linha | cobertura abaixo de 60% |
| `security` | Bandit, pip-audit e Gitleaks | código inseguro, dependência vulnerável ou segredo no histórico |
| `test-report` | resumo dos resultados | nunca falha |
| `pipeline-success` | resultado de todos os jobs | qualquer job anterior não passou |

## Checagens do treino

O `train.yml` só começa depois que o `ci-check` confirma que o CI passou
para o mesmo commit. Depois roda `plan`, `test-end-to-end`, `collect-data`,
`train-autoencoder` e `evaluate`, e termina em `pipeline-success`.

O modelo a treinar é escolhido pelo input `model`, que o job `plan` valida
(o pacote precisa ter `trainer_factory.py` e `overfitting.py`). Com ele, o
treino chama `run_model_training --model <model>` e a avaliação chama
`models/<model>/overfitting`, como nas specs 01, 04 e 06.

O job `structure` do `ci.yml` verifica a estrutura de cada pacote que tem
`trainer_factory.py`, como descrito na spec 01.

Os artifacts do pipeline são `market-data-<ticker>` (preços validados, por
ticker) e `trained-models` (pesos `.pt` e relatórios `*-overfitting.json` de
todos os tickers).

## Definição de pronto

Um modelo novo está pronto quando:

1. segue a estrutura da spec 01;
2. o comando de treino de vários tickers da spec 06 (`run_model_training`)
   e o comando de um ticker da spec 02 (`run_train_script`, se existir)
   passam nos critérios aplicáveis;
3. o relatório da spec 04 é gravado com todas as chaves e passa nos
   critérios CA-01 a CA-04;
4. o pylint passa sem avisos em `src/generative_models` e `tests`;
5. o pytest passa e a cobertura do pacote fica acima de 60%;
6. o treino de ponta a ponta roda com `WANDB_MODE=disabled` e
   `--epochs 2` em um CSV da spec 03;
7. a documentação do README e das specs foi atualizada no mesmo commit,
   como exige o `CLAUDE.md`.

## Fora de escopo

- Ajustes de desempenho e de arquitetura do modelo.
- Deploy do modelo (ele tem workflow próprio, ainda não criado).
