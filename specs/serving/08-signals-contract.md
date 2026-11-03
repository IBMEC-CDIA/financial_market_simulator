# 08 - Contrato de saída do GET /signals

- **Arquivo:** `src/generative_models/serving/signals.py` e rota `GET /signals` em `src/generative_models/serving/api.py`
- **Arquivo de teste:** `tests/generative_models/serving/test_signals.py` e `tests/generative_models/serving/test_api_signals.py`
- **Depende de:** specs 01, 05, 06 e 07
- **Situação:** a implementar

## Objetivo

Garantir que a resposta de `GET /signals` tenha sempre o mesmo formato, para que quem consome a rota (o FinBrain) possa usá-la sem adaptação. A spec fixa nomes de campos, tipos, limites e o valor de retorno quando não há sinal, e os testes verificam cada um deles.

## Origem

- Spec 06: o schema `SignalItem` define os campos e as validações.
- Spec 07: a rota é a única forma pública de obter os sinais. Ela recebe um ticker e um período, como a rota de dados de mercado.
- Regra de conversão de `signals.py`: `default_signal_rule` (proposta, ainda a confirmar pelo time).

## Interface pública

```python
class SignalItem(BaseModel):
    symbol: str
    action: Literal["buy", "sell", "hold"]
    confidence: float
    generated_at: datetime
    model: str | None
    reason: str | None


def build_signals(
    prediction_tables: list[tuple[AutoencoderModelMetadata, pd.DataFrame]],
    rule: SignalRule | None,
    generated_at: datetime,
) -> list[SignalItem]:
    ...


# rota GET /signals
def get_signals(
    service: ModelServiceDependency,
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[SignalItem]:
    ...
```

Parâmetros da rota (query string):

- `ticker` (obrigatório): ticker do modelo, sem diferença entre maiúsculas e minúsculas.
- `start_date` (opcional, `YYYY-MM-DD`): primeiro dia do período. Padrão: 365 dias antes de `end_date`.
- `end_date` (opcional, `YYYY-MM-DD`): último dia do período. Padrão: hoje.

## Formato da resposta

`GET /signals` responde `200` com um array JSON. Cada elemento é um objeto com exatamente estes seis campos, nesta forma:

```json
[
  {
    "symbol": "AAPL",
    "action": "buy",
    "confidence": 0.82,
    "generated_at": "2026-11-02T12:00:00Z",
    "model": "simple-autoencoder-aapl",
    "reason": "Janela de 10 dias com erro de reconstrução acima do limite."
  },
  {
    "symbol": "MSFT",
    "action": "hold",
    "confidence": 0.35,
    "generated_at": "2026-11-02T12:00:00Z",
    "model": "simple-autoencoder-msft",
    "reason": null
  }
]
```

| Campo | Tipo JSON | Regra |
|---|---|---|
| `symbol` | string | Não vazio, em letras maiúsculas. |
| `action` | string | Exatamente `"buy"`, `"sell"` ou `"hold"`. |
| `confidence` | number | Entre 0 e 1, inclusive. |
| `generated_at` | string | ISO 8601 em UTC, terminando em `Z`. |
| `model` | string ou `null` | Até 60 caracteres. `null` quando o nome passa de 60. |
| `reason` | string ou `null` | Até 280 caracteres. Usa apenas números do modelo, sem linguagem de recomendação de investimento. |

## Exemplos de requisição

Base da API: `http://127.0.0.1:8000` em desenvolvimento. Em produção, use a URL pública do serviço, que precisa ter sido publicada com esta versão da API.

```bash
curl -G "http://127.0.0.1:8000/signals"   --data-urlencode "ticker=AAPL"   --data-urlencode "start_date=2026-01-01"   --data-urlencode "end_date=2026-06-30"
```

```javascript
const BASE_URL = "http://127.0.0.1:8000";

async function fetchSignals(ticker, startDate, endDate) {
  const params = new URLSearchParams({ ticker });
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);

  const response = await fetch(`${BASE_URL}/signals?${params}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }
  return response.json();
}

const signals = await fetchSignals("AAPL");
```

Quem consome a rota deve tratar a resposta assim:

- `200`: array JSON, com no máximo um item. Um array vazio é uma resposta válida, não um erro.
- `404` e `422`: corpo JSON com a chave `detail`. No `404` e no `422` de período invertido, `detail` é um texto. No `422` de validação do FastAPI (por exemplo, `ticker` ausente), `detail` é uma lista de erros. O frontend deve mostrar uma mensagem própria ao usuário final, e usar `detail` só para depuração.

Exemplo de `200`:

```json
[
  {
    "symbol": "AAPL",
    "action": "buy",
    "confidence": 0.6,
    "generated_at": "2026-11-03T14:28:46.333568Z",
    "model": "simple-autoencoder-aapl:local",
    "reason": "Janela de 10 dias com erro de reconstrução acima do limite."
  }
]
```

Exemplo de `404`:

```json
{ "detail": "No model for ticker 'TSLA' in '<diretório do servidor>'." }
```

O caminho do diretório do servidor aparece no `detail` do `404`. Essa mensagem é para depuração. O frontend não deve exibi-la ao usuário final.

## Integração com o frontend

- **CORS:** a API não configura CORS. Um frontend em outra origem (outra porta ou domínio) não consegue chamar `/signals` direto do navegador. Antes de integrar, o time precisa decidir: adicionar `CORSMiddleware` com as origens permitidas, ou chamar a API por um servidor intermediário. Esta spec não adiciona CORS, porque o consumidor principal (FinBrain) não precisa dele.
- **Escopo de dados:** cada chamada baixa os preços do período do ticker informado. Períodos longos deixam a resposta mais lenta, por isso o frontend deve pedir apenas o período necessário.

## Comportamento

- A resposta é um array, mesmo quando vazio. Nunca é `null` e nunca é um objeto que envolve o array.
- Não há campos extras. Campos novos exigem uma nova versão desta spec.
- Todos os itens de uma mesma resposta têm o mesmo `generated_at`.
- A rota trata um único ticker por chamada, então a lista tem no máximo um item.
- Um ticker sem decisão ou sem dados suficientes no período responde `[]`.
- Sem regra definida (`rule` igual a `None`), a resposta é `[]` e nenhum preço é baixado.

## Erros esperados

| Situação | Resposta | Mensagem (trecho) |
|---|---|---|
| Ação fora de `buy`, `sell`, `hold` | Erro de validação do schema (`ValidationError`) | `action` |
| `confidence` fora de 0 a 1 | Erro de validação do schema | `confidence` |
| `generated_at` sem fuso | Erro de validação do schema | `generated_at` |
| `reason` com mais de 280 caracteres | Erro de validação do schema | `reason` |
| Ticker sem modelo carregado | `404` (`LookupError`) | `No model for ticker '<TICKER>' in '<diretório>'.` |
| `start_date` igual ou depois de `end_date` | `422` (`ValueError`) | `start_date must be earlier than end_date.` |
| Ticker sem `ticker` na requisição | `422` (validação do FastAPI) | `ticker` |
| Período com poucos preços | `[]`, registrado em log de nível INFO | `Skipping the signal of ...` |

## Critérios de aceite

Cada critério deve virar ao menos um teste.

- **CA-01:** a resposta de `GET /signals` é um array JSON, vazio quando não há sinal.
- **CA-02:** cada item tem exatamente os campos `symbol`, `action`, `confidence`, `generated_at`, `model` e `reason`.
- **CA-03:** `action` aceita somente `buy`, `sell` e `hold`.
- **CA-04:** `confidence` fica entre 0 e 1, com 0 e 1 aceitos.
- **CA-05:** `symbol` é convertido para maiúsculas.
- **CA-06:** `generated_at` é serializado em UTC com `Z` ou `+00:00`, e um horário sem fuso é recusado.
- **CA-07:** `model` vira `null` quando o nome tem mais de 60 caracteres, e `reason` é recusado acima de 280.
- **CA-08:** sem regra, a resposta é `[]` e a coleta de preços não é chamada.
- **CA-09:** um período com preços insuficientes responde `[]`, sem erro.
- **CA-09b:** `ticker` é obrigatório, e `start_date` e `end_date` são opcionais com os padrões descritos acima.
- **CA-09c:** um ticker sem modelo responde `404`, e um período invertido responde `422`.
- **CA-10:** o schema OpenAPI de `SignalItem` lista os seis campos como propriedades e `symbol`, `action`, `confidence` e `generated_at` como obrigatórios.
- **CA-11:** a rota continua respondendo `200` com `GET /models/loaded`, ou seja, nenhum endpoint existente muda.

## Fora de escopo

- Não define a regra de negócio final. A regra padrão é uma proposta, confirmada pelo time em outra decisão.
- Não adiciona autenticação, cache ou persistência dos sinais.
- Não configura CORS. Quando o frontend precisar, a decisão de CORS é uma spec própria.
- Não descreve versionamento de API. Mudança incompatível neste formato exige revisão desta spec e de quem consome a rota.
