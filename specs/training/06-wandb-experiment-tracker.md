# 06 - Rastreamento com wandb

- **Arquivo:** `src/generative_models/tracking/wandb_tracker.py`
- **Arquivo de teste:** `tests/generative_models/tracking/test_wandb_tracker.py`
- **Depende de:** nenhuma
- **Situação:** fornecido. O módulo e os testes já existem no repositório;
  use esta spec e os testes como referência.

## Objetivo

Centralizar em uma classe o ciclo de vida de um experimento no
Weights & Biases: login, início do run, registro de métricas,
registro do modelo treinado como artifact e encerramento. O treino
precisa funcionar mesmo sem chave de API ou sem internet.

## Origem no notebook

- Células 7-8: leitura de `WANDB_API_KEY` (dos `Secrets` do Colab ou
  de um `.env`). No pacote vale só o caminho do `.env`.
- Célula 28: criação do `WandbExperimentTracker` com `project_name` e
  `config`.
- Célula 30: `start_run`, `log_metrics` por época, `log_model` e
  `finish_run`.

## Interface pública

```python
class WandbExperimentTracker:
    def __init__(
        self,
        project_name: str,
        config: dict[str, object],
        env_file_path: str = ".env",
        login_timeout_seconds: int = 10,
    ) -> None:
        ...

    def start_run(self) -> None:
        ...

    def log_metrics(
        self,
        metrics: dict[str, object],
    ) -> None:
        ...

    def log_model(
        self,
        model: nn.Module,
        model_name: str,
        model_file_path: str,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        ...

    def finish_run(self) -> None:
        ...
```

Atributos públicos: `project_name`, `config`, `login_timeout_seconds`,
`wandb_api_key`, `wandb_mode` e `wandb_run`.

## Comportamento

### Construtor

- Carrega o arquivo `env_file_path` com `dotenv.load_dotenv`.
- Lê `WANDB_API_KEY` com `os.getenv` (fica `None` quando não existe).
- Começa com `wandb_mode = "disabled"` e `wandb_run = None`.
- Não faz nenhuma chamada ao wandb.

### `start_run`

1. Tenta o login:
   - com chave: `wandb.login(key=<chave>, timeout=login_timeout_seconds)`;
   - sem chave: `wandb.login(anonymous="allow", timeout=login_timeout_seconds)`;
   - se o login funcionar, `wandb_mode` passa a ser `"online"`;
   - se o login lançar **qualquer** exceção, imprime um aviso e mantém
     `wandb_mode = "disabled"` (a exceção não é propagada).
2. Chama `wandb.init(project=project_name, mode=wandb_mode,
   config=config)` e guarda o retorno em `wandb_run`.
3. Escreve no console uma linha informando o modo e o projeto
   (`tqdm.write`).

### `log_metrics`

- Sempre escreve no console (`tqdm.write`) uma linha no formato
  `nome: valor | nome: valor`, com valores `float` formatados com 6
  casas decimais e os demais tipos convertidos com `str`.
- Se `wandb_run` for `None`, imprime um aviso e não registra nada.
- Caso contrário, chama `wandb_run.log(metrics)`, e não o
  `wandb.log` do módulo.

### `log_model`

- Se `wandb_run` for `None`, imprime um aviso e retorna **sem salvar
  o arquivo**.
- Caso contrário:
  1. salva `model.state_dict()` em `model_file_path` com
     `torch.save`;
  2. cria `wandb.Artifact(name=model_name, type="model",
     description=description, metadata=...)`, em que os metadados
     são `{**config, **metadata}` (em caso de chave repetida, o valor
     de `metadata` prevalece);
  3. adiciona o arquivo com `artifact.add_file(model_file_path)`;
  4. registra com `wandb_run.log_artifact(artifact)`.

### `finish_run`

- Chama `wandb_run.finish()` se houver run ativo. Sem run ativo, não
  faz nada e não lança erro.

## Erros esperados

Nenhuma exceção é lançada por falha de login: o tracker entra em modo
`"disabled"`. Erros de `torch.save` (por exemplo, diretório
inexistente) são propagados.

## Critérios de aceite

Use um `.env` vazio em `tmp_path`, remova `WANDB_API_KEY` com
`monkeypatch.delenv` e substitua o módulo `wandb` por um mock.

- **CA-01:** sem `WANDB_API_KEY`, o construtor deixa
  `wandb_api_key is None`, `wandb_mode == "disabled"` e
  `wandb_run is None`.
- **CA-02:** sem chave, `start_run` chama `wandb.login` com
  `anonymous="allow"` e `wandb.init` com `mode="online"`.
- **CA-03:** com `WANDB_API_KEY` definida, `start_run` chama
  `wandb.login` com `key=<chave>`.
- **CA-04:** se `wandb.login` lançar uma exceção, `start_run` não
  falha e chama `wandb.init` com `mode="disabled"`.
- **CA-05:** com run ativo, `log_metrics` repassa o dicionário para
  `wandb_run.log`.
- **CA-06:** sem run ativo, `log_metrics` não lança erro e não chama
  `log`.
- **CA-07:** `log_metrics({"epoch": 1, "train/loss_mse": 0.5})`
  escreve no console `epoch: 1 | train/loss_mse: 0.500000`.
- **CA-08:** sem run ativo, `log_model` não cria o arquivo de pesos.
- **CA-09:** com run ativo, `log_model` cria o arquivo de pesos,
  cria um artifact do tipo `"model"` com o nome e a descrição
  informados, adiciona o arquivo e chama `log_artifact`.
- **CA-10:** os metadados do artifact combinam `config` e `metadata`;
  com `metadata=None`, os metadados são iguais a `config`.
- **CA-11:** `finish_run` chama `wandb_run.finish()` com run ativo e
  não lança erro sem run ativo.

## Fora de escopo

- Leitura da chave a partir dos `Secrets` do Colab.
- Download de artifacts (pertence à camada de serving).
- Registro de gráficos ou tabelas no wandb.
