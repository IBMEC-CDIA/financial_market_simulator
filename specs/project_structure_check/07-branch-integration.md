# 07 - Integração de modelos por branch

- **Arquivo:** não há código. Esta spec regula os arquivos que cada branch de modelo pode alterar.
- **Arquivo de teste:** não se aplica.
- **Depende de:** specs 01 a 06
- **Situação:** fornecido pelo projeto

## Objetivo

Permitir que cada aluno adicione o próprio modelo em uma pasta de
`models/` (`convolutional_autoencoder`, `lstm_autoencoder` ou
`gru_autoencoder`, ou outra criada depois) sem que os branches entrem
em conflito quando forem para a `main`.

## Arquivos que um branch de modelo pode alterar

Um branch de modelo altera apenas:

- `src/generative_models/models/<seu_modelo>/`, com todos os módulos da spec 01;
- `tests/generative_models/models/<seu_modelo>/`, com os testes do próprio modelo.

Qualquer outra alteração é feita por um PR separado, aberto para a `main`
antes do PR do modelo.

## Arquivos compartilhados (não alterar em branch de modelo)

| Caminho | Motivo |
|---|---|
| `.github/workflows/` | Os workflows são únicos para todos os modelos. |
| `pyproject.toml`, `uv.lock` | Dependências são compartilhadas. Veja a regra de dependências abaixo. |
| `Dockerfile`, `.dockerignore`, `.gitignore`, `.trivyignore` | Afetam a imagem e a segurança de todos. |
| `README.md`, `CLAUDE.md`, `.github/copilot-instructions.md` | Documentação do projeto. |
| `specs/` | Contratos do projeto. |
| `src/generative_models/data/`, `tracking/`, `registry/`, `serving/`, `common/` | Código compartilhado. |
| `src/generative_models/models/base_autoencoder/` | Modelo de referência, mantido pelo professor. |
| `src/generative_models/models/ticker_model_manager.py` e seus testes | Gerenciador compartilhado (spec 06). |
| `src/generative_models/models/__init__.py` | Lista de pacotes. |
| `src/generative_models/models/run_model_training.py` e `training_options.py` | Comando e opções compartilhados por todos os modelos. |
| `tests/generative_models/models/test_run_model_training.py` | Testes do comando compartilhado. |

## Dependências novas

Um modelo que precisa de uma biblioteca nova não altera `pyproject.toml`
nem `uv.lock` no próprio branch. Ele pede a inclusão ao professor, em um
PR separado para a `main`. Assim o `uv.lock` continua sendo um só, e o
`structure` do CI confirma que ele está sincronizado.

## Documentação do modelo

A regra de documentação do `CLAUDE.md` vale para o modelo dentro da própria
pasta: `src/generative_models/models/<seu_modelo>/README.md`, com o que o
modelo faz, os comandos para treinar e os limites conhecidos. O `README.md`
da raiz e as specs são atualizados pelo professor, porque são arquivos
compartilhados.

## Antes de abrir o PR do modelo

1. Atualizar o branch com a `main`: `git merge origin/main`, ou o rebase,
   para pegar os workflows e o gerenciador mais recentes.
2. Rodar, localmente:
   - `uv run pylint src/generative_models tests`
   - `uv run pytest --ignore=tests/end_to_end`
3. Confirmar que o `git status` mostra apenas arquivos das pastas da
   seção "Arquivos que um branch de modelo pode alterar".

## Depois do merge na `main`

O professor roda o `train.yml` manualmente, com `model=<seu_modelo>` (spec
05), para confirmar que o treino, a avaliação e o `pipeline-success`
passam com o modelo novo.

## Critérios de aceite

- **CA-01:** o PR de um modelo altera somente os caminhos permitidos.
- **CA-02:** o PR de um modelo não altera nenhum arquivo da tabela de
  compartilhados.
- **CA-03:** o branch do modelo está atualizado com a `main` no momento
  do PR.
- **CA-04:** o PR não adiciona dependências em `pyproject.toml` ou
  `uv.lock`.

## Fora de escopo

- Ordem de merge entre alunos. Quando dois modelos chegam juntos, o merge
  é independente, porque as pastas são diferentes.
