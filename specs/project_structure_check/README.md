# Padrão de projeto para novos modelos

Estas specs definem o contrato que qualquer modelo novo, em
`src/generative_models/models/<nome_do_modelo>/`, precisa cumprir para
entrar no pipeline de CI e de treino sem alterar nenhum arquivo `.yml`.

Os modelos de exemplo são `base_autoencoder`, `convolutional_autoencoder`,
`lstm_autoencoder` e `gru_autoencoder`. Os dois últimos ainda não têm
código, mas já seguem o padrão de pacote.

## Ordem das specs

| # | Spec | O que define |
|---|---|---|
| 01 | [Estrutura de pacote](01-package-layout.md) | Pastas, `__init__.py` e espelho dos testes |
| 02 | [Contrato de treino](02-training-cli.md) | Comando `run_train_script`, argumentos e arquivos gravados |
| 03 | [Contrato dos dados](03-price-data-contract.md) | CSV de preços que o treino recebe |
| 04 | [Relatório de overfitting](04-overfitting-report.md) | JSON que o job `evaluate` lê |
| 05 | [Portões de qualidade](05-quality-gates.md) | Checagens do CI e critérios de pronto |
| 06 | [Contrato de treinador](06-model-trainer-contract.md) | Função `build_trainer` de cada modelo e o gerenciador `TickerModelManager` |
| 07 | [Integração por branch](07-branch-integration.md) | Arquivos que cada branch de modelo pode alterar, para o merge na `main` |
| 08 | [Serving e build da imagem](08-serving-and-image.md) | Modelos locais carregados na subida, roteamento por ticker e build com os modelos dentro da imagem |

## Como as specs são usadas

- O job `structure` do `ci.yml` verifica que a pasta `tests/` existe e que
  os arquivos obrigatórios estão no repositório.
- O `train.yml` chama, para o modelo escolhido, os módulos
  `run_model_training` (com `--model`) e `overfitting`. Por isso o contrato das specs 02 a 06
  é o que faz o workflow rodar.
- Cada branch de modelo só altera a própria pasta, como define a spec 07.
- Um modelo novo só é considerado pronto quando cumpre a spec 05.
