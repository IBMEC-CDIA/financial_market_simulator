# Trilha de treino

Converte o treinamento do notebook
`draft/autoencoder_outlier_gab_draft.ipynb` em um pipeline executável
pela linha de comando:

```bash
uv run python -m generative_models.models.autoencoder.train_autoencoder \
    --ticker AAPL
```

O fluxo de trabalho, as regras gerais e a definição de pronto estão
no [README das specs](../README.md).

## Ordem das specs

| # | Spec | Arquivo | Situação | Depende de |
|---|---|---|---|---|
| 01 | [Dados de mercado](01-market-data.md) | `data/market_data.py` | Fornecido | - |
| 02 | [Normalização](02-normalization.md) | `common/normalization.py` | Fornecido | - |
| 03 | [Dataset de janelas deslizantes](03-sliding-window-dataset.md) | `data/datasets.py` | Fornecido | - |
| 04 | [Autoencoder denso](04-simple-autoencoder.md) | `models/autoencoder/autoencoder.py` | A implementar | - |
| 05 | [Configuração do treino](05-training-config.md) | `models/autoencoder/config.py` | A implementar | - |
| 06 | [Rastreamento com wandb](06-wandb-experiment-tracker.md) | `tracking/wandb_tracker.py` | Fornecido | - |
| 07 | [Pipeline de treino](07-autoencoder-trainer.md) | `models/autoencoder/trainer.py` | A implementar | 01 a 06 |
| 08 | [Script de linha de comando](08-train-cli.md) | `models/autoencoder/train_autoencoder.py` | A implementar | 05, 07 |

Os caminhos de **Arquivo** são relativos a `src/generative_models/`.
As specs fornecidas descrevem módulos que o notebook já importa do
pacote; leia-as antes da spec 04, porque as specs 07 e 08 dependem
delas. A implementação começa pela spec 04.

## Mapa notebook -> módulos

| Células do notebook | O que fazem | Spec |
|---|---|---|
| 7-8 | Chave de API do wandb | 06 |
| 12 | Coleta dos preços de fechamento | 01 |
| 16 | Cálculo dos log-retornos | 01, 07 |
| 18 | Separação treino/validação e normalização | 02, 07 |
| 20 | Janelas deslizantes e `DataLoader` | 03, 07 |
| 22-24 | Definição e instanciação do autoencoder | 04, 07 |
| 26 | Escore de reconstrução por janela | 07 |
| 28 | Hiperparâmetros e `config` do wandb | 05, 06 |
| 30 | Laço de treino, validação e registro do modelo | 06, 07 |
| 32 | Limiar de anomalia | 07 |
| - | Execução pela linha de comando | 08 |

As células 33 em diante (alinhamento dos rótulos, sinais de compra e
venda, gráficos e GARCH) estão **fora do escopo** destas specs. A
detecção de outliers em dados novos, com o modelo já treinado, é
tratada na [trilha de serving](../serving/README.md).

## Contrato com a trilha de serving

A trilha de serving carrega os modelos registrados no wandb pela
spec 07. Por isso, três pontos desta trilha não podem mudar:

- o nome do artifact: `simple-autoencoder-<ticker_slug>` (spec 07);
- os metadados do artifact: `config` do run mais `final_train_loss`,
  `validation_loss`, `anomaly_threshold`, `return_train_mean` e
  `return_train_std` (specs 06 e 07);
- a arquitetura e as chaves do `state_dict` do `SimpleAutoencoder`
  (spec 04).
