# Explicação do projeto

Este documento explica o Financial Market Simulator para quem vai ler, rodar,
testar ou estender o projeto. Ele descreve a ideia, as camadas do código, o
ambiente necessário e os passos para executar cada parte. Os detalhes de cada
módulo estão nas specs da pasta [`specs/`](../specs/README.md).

## 1. Ideia do projeto

O projeto usa **autoencoders** para detectar movimentos incomuns no preço de
ativos financeiros. Para cada ticker (por exemplo `AAPL` ou `PETR4.SA`):

1. baixa o preço de fechamento diário;
2. calcula o **log-retorno** diário;
3. treina um autoencoder com janelas de log-retornos normais;
4. usa o **erro de reconstrução** de cada janela como escore de anomalia.

Uma janela que o modelo reconstrói mal é marcada como **anômala**, ou seja, o
mercado se comportou de forma incomum naquele período.

O projeto é de ensino e pesquisa. Os modelos descrevem comportamentos incomuns
em preços passados. Eles não preveem preços e não são recomendação de
investimento. Os resultados dependem do período e do ticker, e não foram
validados para negociação.

## 2. Arquitetura em camadas

```
 Dados               Modelo               Treino e avaliação
 ───────────────     ───────────────      ─────────────────────────────
 yfinance       ->   autoencoder    ->   run_train_script (um ticker)
 validação (GE)      trainer              run_model_training (vários)
 log-retornos        overfitting          TickerModelManager
 split 70/15/15      config               │
 janelas             │                    ├─ wandb (tracking)
                     │                    └─ artefatos: .pt + *-overfitting.json
                     ▼
 Entrega             Serving                    Registro
 ─────────────       ─────────────────────      ─────────────────────────────
 GitHub Actions  ->  FastAPI (api.py)     <-    models/ (local)
 Docker image        ModelService               wandb (download_latest_models)
 Docker Hub          Predictor, signals
 Azure ACI           /predictions, /signals
```

### 2.1 Dados (`src/generative_models/data/`)

- `market_data.py` e `collect_market_data.py`: baixam o preço de fechamento
  pelo yfinance.
- `validation.py`: checa a série com o Great Expectations antes do uso.
- `normalization.py`: normaliza os log-retornos.
- `datasets.py` e `price_dataset.py`: dividem os dados em treino (70%),
  validação (15%) e teste (15%), em ordem de tempo, e criam janelas deslizantes
  de log-retornos.

O conjunto de teste é avaliado uma única vez, depois do treino.

### 2.2 Modelo (`src/generative_models/models/`)

O pacote `base_autoencoder` é o modelo de referência do projeto:

- `autoencoder.py`: a rede densa, com camada oculta e camada latente.
- `trainer.py`: o laço de treino (Adam, perda de reconstrução).
- `overfitting.py`: verifica se o modelo supera a linha de base ingênua e se
  o erro de validação não está muito acima do erro de treino.
- `config.py`: dataclasses com os hiperparâmetros padrão.
- `trainer_factory.py`: função `build_trainer`, que segue o contrato descrito
  em [spec 06](../specs/project_structure_check/06-model-trainer-contract.md).
- `run_train_script.py`: linha de comando para treinar um ticker.

Hiperparâmetros padrão (de `config.py`):

| Parâmetro | Valor padrão |
|---|---|
| Janela (`window-size`) | 10 log-retornos |
| Camada oculta / latente | 16 / 4 |
| Lote (`batch-size`) | 32 |
| Épocas | 300 |
| Taxa de aprendizado | 0,001 (Adam) |
| Semente | 42 |
| Contaminação | 0,05 (define o limiar de anomalia) |

O limiar de anomalia é calculado a partir da contaminação esperada. Ele é
salvo junto com os pesos do modelo.

Os pacotes `autoencoder`, `convolutional_autoencoder`, `gru_autoencoder` e
`lstm_autoencoder` existem como placeholders, com apenas um `__init__.py`
com docstring. O pacote `regression` tem um modelo de regressão feed-forward.

### 2.3 Treino em lote (`models/run_model_training.py`, `ticker_model_manager.py`)

- `TickerModelManager` treina o modelo de vários tickers em sequência. Para
  cada um, baixa e valida os preços, cria o treinador com a função
  `build_trainer` do pacote escolhido e executa o treino.
- `run_model_training.py` é o comando compartilhado que o workflow `train.yml`
  chama. Um modelo novo não deve copiar essa lógica, conforme a
  [spec 06](../specs/project_structure_check/06-model-trainer-contract.md).

### 2.4 Tracking (`src/generative_models/tracking/`)

`WandbExperimentTracker` (em `wandb_tracker.py`) registra os experimentos no
Weights & Biases: configuração, métricas por época e o modelo como artefato. A
chave vem da variável `WANDB_API_KEY`, lida de um arquivo `.env` local. Sem a
chave, o treino continua com o tracking desativado.

### 2.5 Registro de modelos (`src/generative_models/registry/`, `serving/`)

Um modelo treinado é salvo como:

- `<output-directory>/simple-autoencoder-<ticker>.pt`: os pesos e o limiar;
- `<output-directory>/simple-autoencoder-<ticker>-overfitting.json`: o
  relatório de overfitting e de linha de base.

Há dois jeitos de carregar os modelos no serviço:

- **Local:** `local_autoencoder_registry.py` lê a pasta `models/`, indicada
  pela variável `AUTOENCODER_MODEL_DIR`.
- **wandb:** `wandb_autoencoder_registry.py` e `download_latest_models.py`
  baixam a versão de cada modelo registrado no wandb sob o alias do
  experimento (`latest` por padrão).

### 2.6 Serving (`src/generative_models/serving/`)

Uma API FastAPI (`api.py`) carrega os modelos e responde às requisições:

| Método | Rota | Função |
|---|---|---|
| GET | `/health` | Verifica se o serviço está no ar. |
| GET | `/models/loaded` | Lista os modelos em memória. |
| GET | `/models/available` | Lista os modelos disponíveis no wandb. |
| GET | `/models/default` | Mostra o modelo padrão. |
| POST | `/models/reload` | Baixa novamente um modelo. |
| POST | `/predictions/log-returns` | Detecta anomalias em log-retornos. |
| POST | `/predictions/prices` | Detecta anomalias em preços de fechamento. |
| POST | `/predictions/market` | Detecta anomalias em dados de mercado baixados na hora. |
| GET | `/signals` | Sugestão de compra, venda ou manutenção de um ativo. |

A rota `/signals` usa a regra `default_signal_rule` de
`generative_models.serving.signals`. Ela é uma proposta que a equipe ainda
precisa confirmar. Os formatos de entrada e saída estão nas specs
[07](../specs/serving/07-api.md) e [08](../specs/serving/08-signals-contract.md).

Quando a variável `AUTOENCODER_MODEL_DIR` está definida, os modelos vêm da
pasta local e o wandb não é usado.

## 3. Estrutura de pastas

```
financial_market_simulator/
├── .github/
│   ├── actions/setup-python-uv/   # ação composta: Python 3.12 e uv
│   └── workflows/                 # ci.yml, train.yml, build.yml, deploy.yml
├── docs/                          # esta explicação
├── models/                        # modelos treinados usados pela imagem
├── notebooks/                     # notebooks de exploração e ensino
├── specs/                         # especificações dos módulos
├── src/generative_models/         # código da biblioteca
│   ├── data/
│   ├── models/
│   ├── registry/
│   ├── serving/
│   └── tracking/
├── tests/                         # espelha a estrutura de src/
├── Dockerfile                     # imagem da API
├── pyproject.toml                 # dependências e configuração
├── requirements.txt               # ambiente completo de desenvolvimento
└── uv.lock                        # versões travadas pelo uv
```

## 4. Ambiente

### 4.1 Pré-requisitos

- [uv](https://docs.astral.sh/uv/), o gerenciador de pacotes e de ambientes.
- Python 3.12. O uv instala a versão automaticamente, se preciso.
- Docker, apenas para construir e rodar a imagem da API.
- Uma chave do Weights & Biases, se o tracking for usado.

### 4.2 Variáveis de ambiente

| Variável | Onde | Para que serve |
|---|---|---|
| `WANDB_API_KEY` | `.env` local, ou secret no GitHub | Login no wandb. |
| `AUTOENCODER_MODEL_DIR` | ambiente da API | Pasta com os modelos. A imagem usa `/app/models`. |
| `AUTOENCODER_WANDB_PROJECT` | ambiente da API | Projeto do wandb para baixar os modelos. |
| `AUTOENCODER_WANDB_ENTITY` | ambiente da API | Usuário ou time do wandb. |
| `AUTOENCODER_DEVICE` | ambiente da API | `cpu` ou `cuda`. |
| `AUTOENCODER_PRELOAD_MODEL` | ambiente da API | `false` para não carregar os modelos na inicialização. |

Nunca coloque a chave do wandb no código. O `.env` está no `.gitignore` e não
deve ser versionado.

### 4.3 Dependências

- `pyproject.toml` declara as dependências do pacote: NumPy, Pandas, PyTorch,
  wandb, python-dotenv, yfinance, tqdm, FastAPI e Uvicorn. O grupo `dev`
  acrescenta pytest, pytest-cov, pylint, httpx e Great Expectations.
- `requirements.txt` é o ambiente completo usado localmente. Ele inclui
  Jupyter, Plotly e torchinfo, e aponta o PyTorch para o índice cu121, com `torch==2.5.1+cu121`.
- `uv.lock` trava as versões. O Docker usa `uv sync --frozen`.

## 5. Como rodar

### 5.1 Instalar o ambiente

```bash
uv venv --python 3.12.10
uv sync --extra dev
```

Para o ambiente completo de desenvolvimento, com notebooks:

```bash
uv pip install -r requirements.txt
```

Crie o arquivo `.env` na raiz com a chave do wandb:

```text
WANDB_API_KEY=<sua-chave>
```

### 5.2 Rodar os testes

```bash
uv run pytest
```

### 5.3 Treinar um modelo

Exemplo rápido, com poucos dados e duas épocas (PowerShell):

```powershell
uv run python -m generative_models.models.base_autoencoder.run_train_script `
    --ticker AAPL `
    --start-date 2024-01-01 `
    --end-date 2024-12-31 `
    --epochs 2 `
    --device cpu
```

O script salva `artifacts/simple-autoencoder-aapl.pt`. Para ver todas as
opções:

```bash
uv run python -m generative_models.models.base_autoencoder.run_train_script --help
```

### 5.4 Servir a API localmente

Copie os arquivos do modelo para `models/` (o `.pt` e o `*-overfitting.json`) e
suba a API apontando para essa pasta:

```powershell
$env:AUTOENCODER_MODEL_DIR = "models"
uv run uvicorn generative_models.serving.api:app --host 127.0.0.1 --port 8000
```

Depois, abra `http://127.0.0.1:8000/docs` para a documentação interativa, ou
`http://127.0.0.1:8000/health` para checar o serviço.

### 5.5 Construir e rodar a imagem Docker

```bash
docker build -t financial-market-simulator .
docker run --rm -p 8000:8000 financial-market-simulator
```

A imagem inclui a pasta `models/`. Por isso, antes de construí-la, ela precisa
conter os modelos treinados. O workflow `build.yml` cuida disso, como descrito
na próxima seção.

## 6. Pipeline de entrega (GitHub Actions)

| Workflow | Quando roda | O que faz |
|---|---|---|
| `ci.yml` | Push e pull request | Estrutura do projeto, lint, testes unitários, análise de segurança e relatório. |
| `train.yml` | Manual | Confere o CI do commit, valida o modelo e os tickers, roda os testes de ponta a ponta, coleta os preços, treina e avalia o overfitting. |
| `build.yml` | Manual | Baixa os modelos (de um run de treino ou do experimento informado no wandb), constrói a imagem, testa a API, escaneia vulnerabilidades com Trivy e envia a imagem ao Docker Hub. |
| `deploy.yml` | Manual | Substitui o container no Azure a partir de uma imagem do Docker Hub, e espera a API responder com um modelo carregado. |

Configuração necessária no GitHub (*Settings > Secrets and variables > Actions*):

| Nome | Tipo | Usado por |
|---|---|---|
| `WANDB_API_KEY` | secret | `train.yml`, `build.yml` |
| `DOCKERHUB_USERNAME` | secret | `build.yml`, `deploy.yml` |
| `DOCKERHUB_TOKEN` | secret | `build.yml`, `deploy.yml` |
| `AZURE_CREDENTIALS` | secret | `deploy.yml` |
| `AZURE_RESOURCE_GROUP` | variável | `deploy.yml` |
| `AZURE_LOCATION` | variável, opcional (padrão `brazilsouth`) | `deploy.yml` |
| `DEPLOY_ENVIRONMENT` | variável, opcional (padrão `production`) | `deploy.yml` |

A ordem típica de uso é: treinar (`train.yml`), construir e publicar a imagem
(`build.yml`) e colocá-la no ar (`deploy.yml`). O README tem os comandos
para criar o service principal do Azure.

## 7. Limites e cuidados

- A saída do modelo é um escore de anomalia, não uma previsão de preço.
- A regra de sinais é uma proposta, ainda sem confirmação da equipe.
- O deploy não tem rollback. Se a criação falhar depois de apagar o container
  antigo, o serviço fica fora do ar até um novo deploy.
- Resultados mudam com o período e o ticker. Nenhum resultado foi validado
  para negociação.

## 8. Onde encontrar mais

- [`README.md`](../README.md): visão geral e referência dos workflows.
- [`specs/`](../specs/README.md): especificações de cada módulo, usadas
  também nas atividades do curso.
- [`CLAUDE.md`](../CLAUDE.md) e
  [`.github/copilot-instructions.md`](../.github/copilot-instructions.md):
  convenções de código, testes e documentação.
