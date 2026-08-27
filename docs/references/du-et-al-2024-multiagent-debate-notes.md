# Reading notes: multiagent debate for factuality and reasoning

Reference: Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor
Mordatch. *Improving Factuality and Reasoning in Language Models through
Multiagent Debate*. ICML 2024.

- [OpenReview paper and metadata](https://openreview.net/forum?id=zj7YuTE4t8)
- [Authors' implementation](https://github.com/composable-models/llm_multiagent_debate)

These notes summarize the parts of the paper that define this repository's
reproduction target. They are not a substitute for the paper.

## Core method

The paper samples multiple language-model agents independently, then repeats a
review step in which each agent sees answers from its peers and updates its own
response. A final answer is selected from the agents after a fixed number of
rounds or after their answers converge.

For the reasoning experiments, the important control variables are:

- number of agents;
- number of debate rounds;
- model family;
- whether peer responses contain full reasoning or only final answers;
- the aggregation rule used to convert agent answers into a group prediction.

The reported experiments cover arithmetic, grade-school mathematics, chess
move selection, factual generation, and other reasoning tasks. The authors also
test heterogeneous models, fixed specialty roles, summarized peer context, and
the relationship between agreement and correctness.

## Findings relevant to this project

### More agents or rounds are not a universal guarantee

The paper reports gains in selected controlled settings, including agent-count
and round-count sweeps. Those curves are empirical results for particular
models, prompts, and tasks; they do not establish monotonic improvement for
every model. Reproductions should therefore report the complete sweep, cost,
and failure rate rather than assume that additional debate is beneficial.

### Full reasoning can help peer review

Agents need enough information to identify a peer's mistake. The paper finds
that sharing reasoning can outperform sharing final answers alone, while
summarization may reduce context cost without necessarily reducing quality.
This motivates keeping protocol format as an explicit benchmark variable.

### Agreement and correctness are different

Debate can increase consensus while preserving or spreading a wrong answer.
The paper's discussion of agreeableness makes consensus, false consensus, and
answer-transition direction necessary companion metrics to final accuracy.

### Specialization already has a baseline

The paper includes fixed domain-specialist personas and heterogeneous-model
experiments. Checker roles in this repository are therefore framed as an
ablation of verification behavior, not as a claim that role assignment itself
is novel.

### Cost and context are first-order variables

Every additional agent and turn adds model calls and peer context. Latency,
token usage, completion rate, and context truncation must be reported with
accuracy because they can change which protocols are practical and can also
confound the observed result.

## Reproduction mapping

| Paper behavior | Workbench implementation |
| --- | --- |
| Independent initial answers | One first-turn response per agent with no peer context |
| Repeated peer review | Each later turn receives peers' latest full responses |
| Agent and round sweeps | Configurable 1–7 agents and 1–4 total turns |
| Reasoning benchmarks | Arithmetic, GSM8K, and SVAMP adapters |
| Group answer | Paper-compatible plurality plus strict-majority diagnostics |
| Full reasoning context | Plain-text conversation history in reproduction mode |
| Heterogeneity ablation | Optional dedicated arithmetic or semantic checker |

The structured Web protocol is intentionally separate. It adds JSON schemas,
task roles, research tools, claims, directed targets, persistence, streaming,
and final synthesis for inspection. Results from that protocol should not be
described as exact paper reproductions.

## Questions carried into the experiments

1. Does the paper's qualitative scaling behavior persist with a much smaller,
   quantized local model?
2. Does assigning one agent an explicit verification duty improve the balance
   between corrected errors and newly introduced errors?
3. Can a checker resist persuasive but incorrect peer context over several
   synchronous turns?
4. How quickly do latency and context cost grow relative to any accuracy gain?

The initial results are reported in
[`../reports/small-model-debate-failure-analysis.md`](../reports/small-model-debate-failure-analysis.md).
