# makefun-free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/agniakhrli/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/agniakhrli/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A single Node 22 server that unifies ten simulated free-tier AI providers (OpenAI, Anthropic and Gemini wires) behind one OpenAI-compatible and one Anthropic-compatible endpoint. It reads the trial brief, probes each provider's catalog, ranks the free models that meet each request by quality rating, spreads calls across providers under their per-trial budgets, fails over to the next best free model on 402/429/5xx, and translates requests and streaming events across wires — deterministically, so a live trial and its replay leave the same state export. Practice trial-0 score 77/100 (catalog 100, failover 83, quality 71, served 80, wire 93).

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kmb7c9xk&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
