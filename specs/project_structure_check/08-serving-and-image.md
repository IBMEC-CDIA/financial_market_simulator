# 08 - Serving com modelos locais e build da imagem

- **Arquivo:** `src/generative_models/serving/local_autoencoder_registry.py`,
  `src/generative_models/serving/model_service.py`,
  `src/generative_models/serving/api.py`, `Dockerfile` e
  `.github/workflows/build.yml`
- **Arquivo de teste:** `tests/generative_models/serving/test_local_autoencoder_registry.py`
  e `tests/generative_models/serving/test_api_local_routing.py`
- **Depende de:** specs 04 e 06
- **Situação:** implementada

## Objetivo

Servir os modelos treinados a partir de uma pasta local, com todos os
modelos carregados em memória na subida da API. Cada predição é roteada
para o modelo do ticker da requisição. A imagem Docker leva os modelos
dentro dela, e por isso não precisa da chave do wandb para servir.

## Pasta de modelos

A pasta indicada por `AUTOENCODER_MODEL_DIR` contém, para cada modelo, dois
arquivos com o mesmo nome:

- `<nome>-overfitting.json`: o relatório da spec 04, com os campos de
  serving (`window_size`, `hidden_layer_size`, `latent_layer_size`,
  `return_train_mean`, `return_train_std`, `anomaly_threshold`) e `ticker`;
- `<nome>.pt`: os pesos do modelo.

Um relatório sem o `.pt` correspondente, ou sem algum campo de serving, é
ignorado.

## Comportamento

1. Na subida, a API chama `preload_all()`, que carrega todos os modelos da
   pasta em memória. Depois carrega o modelo padrão.
2. `POST /predictions/log-returns` e `POST /predictions/prices` usam o campo
   `ticker` para escolher o modelo na memória. Um ticker sem modelo gera
   `404`.
3. `GET /models/loaded` lista os modelos carregados, com o ticker e a janela.
4. Com `AUTOENCODER_PRELOAD_MODEL=false`, nenhum modelo é carregado na
   subida; cada predição carrega o modelo do ticker na primeira vez.

## Build da imagem (`build.yml`)

1. O job `ci-check` confirma que o CI passou para o commit. Se o CI do commit
   ainda estiver em execução, o job espera até 30 minutos. Ele falha se o CI
   terminar sem sucesso, ou se nenhum run aparecer em 3 minutos.
2. O job `build-image` preenche a pasta `models/` de uma destas formas:
   - com `train_run_id` informado, baixa o artifact `trained-models` desse
     run do `train.yml`;
   - sem `train_run_id`, baixa a versão `latest` do artifact
     `simple-autoencoder-<ticker>` de cada ticker do input `tickers`, pelo
     módulo `generative_models.serving.download_latest_models`. Esse modo
     precisa da chave do wandb (input `wandb_api_key` ou secret
     `WANDB_API_KEY`). O módulo grava `<nome>.pt` e
     `<nome>-overfitting.json`, que o registro local lê. Os campos
     `test_loss` e `baseline_test_loss` ficam `null`, porque o artifact do
     wandb não os guarda.
     O comportamento completo está na
     [spec 09 da trilha de serving](../serving/09-latest-model-download.md).
3. A imagem é construída. O Dockerfile copia `models/` para `/app/models` e
   define `AUTOENCODER_MODEL_DIR=/app/models`.
4. Um teste de fumaça sobe o container e confere `/health`, que
   `/models/loaded` tem pelo menos um modelo e que uma predição com janela
   de zeros responde.
5. O Trivy escaneia a imagem, e o job falha em CVEs HIGH ou CRITICAL com
   correção.

6. O job `push-image` baixa a imagem do job `build-image`, faz login no
   Docker Hub e envia as tags `<usuario>/financial-market-simulator:<sha>` e
   `:latest`. O usuário e o token vêm dos inputs `dockerhub_username` e
   `dockerhub_token`, e na falta deles, dos secrets `DOCKERHUB_USERNAME` e
   `DOCKERHUB_TOKEN`. O job falha se algum dos dois estiver ausente.

## Erros esperados

| Situação | Resposta |
|---|---|
| Ticker sem modelo na pasta | `404` |
| Janela de log-retornos menor que a do modelo | `422` |
| Pasta sem modelos completos | a API sobe com a lista vazia; `/models/loaded` retorna `[]` |
| Sem `train_run_id` e sem `tickers` | o job `build-image` falha antes do download |
| Sem `train_run_id` e sem chave do wandb | o job `build-image` falha antes do download |
| Ticker sem artifact `latest` no wandb | o job `build-image` falha antes do build |
| `train_run_id` de um run sem o artifact `trained-models` | o job `build-image` falha |

## Critérios de aceite

- **CA-01:** com dois modelos na pasta, `/models/loaded` lista os dois
  tickers depois da subida.
- **CA-02:** uma predição com o ticker `MSFT` usa o modelo de `MSFT`.
- **CA-03:** um ticker sem modelo responde `404`.
- **CA-04:** um relatório sem o `.pt` correspondente não é listado.
- **CA-05:** a imagem leva `models/` e define `AUTOENCODER_MODEL_DIR`. A
  chave do wandb só é usada no modo sem `train_run_id`, e somente para
  baixar os modelos antes do build.
- **CA-06:** o modo sem `train_run_id` grava, para cada ticker, o `.pt` e o
  `-overfitting.json` com os campos que o registro local exige.

## Limites

- O serving reconstrói apenas o `SimpleAutoencoder` do pacote
  `base_autoencoder`. Um modelo de outro pacote precisa de um leitor próprio.
- Os artifacts do GitHub expiram conforme a retenção do repositório. Um
  build com um `train_run_id` antigo pode falhar.
- Trocar os modelos exige um novo build da imagem.
