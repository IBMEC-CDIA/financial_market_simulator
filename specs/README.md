# Specs do projeto

Esta pasta descreve, módulo a módulo, como o notebook
`draft/autoencoder_outlier_gab_draft.ipynb` é convertido no pacote
Python `src/generative_models/`: primeiro o treinamento completo do
`SimpleAutoencoder` e, depois, a disponibilização do modelo treinado
para inferência (script de predição e API REST).

Cada spec define **o contrato** de um módulo (interface pública,
comportamento, erros esperados e critérios de aceite). Ela não diz
**como** implementar: essa decisão é sua, desde que todos os critérios
de aceite sejam atendidos.

## Trilhas

| Trilha | Pasta | O que entrega | Pré-requisito |
|---|---|---|---|
| Treino | [`training/`](training/README.md) | Pipeline de treino executável pela linha de comando, com rastreamento no wandb | - |
| Serving | [`serving/`](serving/README.md) | Carregamento dos modelos do wandb, script de predição e API FastAPI | Trilha de treino concluída |

Cada pasta tem um `README.md` com a ordem das specs e o mapa entre as
células do notebook (ou os módulos de treino) e os módulos a
implementar.

## Situação das specs

Cada spec traz, no cabeçalho, uma das situações abaixo:

| Situação | Significado | O que fazer |
|---|---|---|
| **A implementar** | O módulo e os testes não existem no repositório. | Siga o fluxo de trabalho completo. |
| **Fornecido** | O módulo e os testes já existem, porque o notebook de partida já os importa do pacote. | Leia a spec e os testes existentes como referência; não reimplemente. |

Os módulos fornecidos são `data/market_data.py`, `data/datasets.py`,
`common/normalization.py`, `tracking/wandb_tracker.py` e
`serving/model_registry.py`. Eles mostram, em código já pronto, o
padrão de implementação e de testes esperado nas specs a implementar.

Dentro de uma trilha, uma referência como "spec 05" aponta para a
spec 05 da mesma pasta. Referências entre trilhas usam o nome da
pasta, por exemplo `training/07` ou `serving/03`.

## Fluxo de trabalho

Para cada spec **a implementar**, na ordem do `README.md` da trilha:

1. Leia a spec inteira e localize a origem indicada em **Origem**
   (células do notebook ou módulos já implementados).
2. Escreva os testes unitários no arquivo indicado em **Arquivo de
   teste**, com pelo menos um teste por critério de aceite (`CA-xx`).
   Cite o código do critério na docstring do teste.
3. Rode os testes e confirme que eles falham (o módulo ainda não
   existe).
4. Implemente o módulo no arquivo indicado em **Arquivo**.
5. Rode `uv run pylint src/generative_models tests` e
   `uv run pytest` até ficarem verdes.
6. Abra um pull request com a spec implementada. A CI
   (`.github/workflows/autoencoder_pipeline.yml`) executa as mesmas verificações.

Só avance para a próxima spec quando a anterior estiver verde: cada
módulo depende dos anteriores.

Os testes ficam em `tests/generative_models/`, espelhando a estrutura
de `src/generative_models/` (por exemplo, `data/datasets.py` é testado
em `tests/generative_models/data/test_datasets.py`). Lembre de criar
os arquivos `__init__.py` dos pacotes, tanto em `src/` quanto em
`tests/`.

## Regras gerais

Valem para todas as specs, além do que cada uma define:

- Siga as convenções de código de [`CLAUDE.md`](../CLAUDE.md):
  docstrings completas em inglês no estilo numpydoc (`Parameters`,
  `Returns`, `Raises`, `Example`), type hints em parâmetros e
  retornos, assinaturas com 2 ou mais parâmetros quebradas em uma
  linha por parâmetro, nomes descritivos em inglês e nenhum
  comentário no código.
- Os testes não podem acessar a internet nem o wandb. Substitua
  `fetch_price_series`, `yfinance.download`, `wandb` e `wandb.Api`
  por mocks (`unittest.mock.patch`) e use `tmp_path` para arquivos.
- Os testes que envolvem modelos devem rodar em CPU e em poucos
  segundos: use séries sintéticas curtas, poucas épocas e
  autoencoders sem treino quando o valor dos pesos não importar.
- Nenhuma chave de API pode aparecer no código. A chave do wandb é
  lida da variável `WANDB_API_KEY` de um arquivo `.env` local.

## Definição de pronto

Uma spec está concluída quando:

- todos os critérios de aceite têm ao menos um teste correspondente;
- `uv run pytest` e `uv run pylint src/generative_models tests`
  passam sem erros;
- a interface pública (nomes, parâmetros, tipos e valores padrão) é
  exatamente a descrita na spec, para que as specs seguintes
  funcionem sobre ela.

Para escrever uma spec nova, use o [template](_template.md).
