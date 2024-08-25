# LogBatcher
Repository for the paper: Demonstration-Free: Towards More Practical Log Parsing with Large Language Models

## Work Flow
![workflow](outputs/figures/workflow.png)

In this work, we propose LogBatcher, a cost-effective LLM-based log parser that requires no training process or labeled data.
Log Batcher contians three main components: **Partitioning, Caching and Batching - Querying** 

## Setup

### 1.Library and Config
To satrt with LogBatcher, you need....

Install all library:
```bash
$ pip install -r requirements.txt
```
Upload your API Key in `config.json`:
```json
{
    "api_key_from_openai": "Your API Key from OpenAI"
}
```
### 2.Execution with Arguments

- To evaluate on smaller dataset with LogBatcher, execute:

```bash
python evaluation_2k.py --batch_size [batch size] --sampling_method [sampling method] --model [model]
```

- To perform online parsing on bigger dataset, add your log file to `dataset` and execute:
```shell
python evaluation_full.py --batch_size [batch size] --chunk_size [chunk size] --sampling_method [sampling method] --model [model]
```

The parsed result is stored in `outputs/parser`, along with results of evaluation metric.