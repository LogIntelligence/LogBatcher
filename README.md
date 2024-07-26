# LogBatcher
Repository for the paper: Demonstration-Free: Towards More Practical Log Parsing with Large Language Models
## Additional Results
### A1. Performance of LILAC with different numbers of demonstration

<div style="text-align: center;">

|   **LILAC**  |   **GA**  |  **MLA**  |   **ED**  |  **FGA**  |  **FTA**  |
|:------------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| **32 shots** |    0.909  |    0.812  |    0.972  |    0.951  |    0.839  |
| **16 shots** |    0.859  |    0.776  |    0.952  |    0.939  |    0.808  |
|  **8 shots** |    0.876  |    0.825  |    0.963  |    0.927  |    0.768  |
|  **4 shots** |    0.845  |    0.766  |    0.942  |    0.915  |    0.753  |
|  **2 shots** |    0.877  |    0.776  |    0.950  |    0.904  |    0.724  |
|  **1 shots** |    0.836  |    0.725  |    0.930  |    0.904  |    0.708  |
|  **0 shots** |    0.765  |    0.601  |    0.903  |    0.824  |    0.577  |

</div>

![A1](outputs/figures/A1.png)

### A2. Performance of LogBatcher over 5 iterations

<div style="text-align: center;">

| **LoaBatcher** | **GA** | **MLA** |  **ED** | **FGA** | **FTA** |
|:--------------:|:------:|:-------:|:-------:|:-------:|:-------:|
|      **Test 1**     |  0.974 |  0.905  | 0.972 |   0.930  |  0.803  |
|      **Test 2**     |  0.974 |  0.898  | 0.971 |   0.930  |  0.796  |
|      **Test 3**     |  0.969 |  0.899  | 0.969 |  0.931  |  0.801  |
|      **Test 4**     |  0.974 |  0.914  | 0.974 |   0.93  |  0.808  |
|      **Test 5**     |  0.980  |  0.906  | 0.973 |  0.934  |  0.803  |

</div>

### A3. Performance of LogBatcher on larger datasets

## (a)
Examples of nconsistent labels across log data in loghub-2.0 is shown below:
```
In dataset Linux and Thunderbird, there are similar logs contains:
`session opened for user cyrus by (uid=0)` and `session opened for user root by LOGIN(uid=0)`
In Linux the label of them is `session opened for user <*> by <*>(uid=<*>)`, while In Thunderbird it is `session opened for user <*> by <*>`
It raises another question about whether placeholder represents null value should appear:
Previous work labeled `connection from 84.139.180.196 (p548BB4C4.dip0.t-ipconnect.de) at Fri Jan 6 15:53:55 2006` and `connection from 84.139.180.196 () at Fri Jan 6 15:53:55 2006` into:
`connection from <*> (<*>) at <*>` and `connection from <*> () at <*>`, while loghub-2.0 use the same label `connection from <*> (<*>) at <*>` to cover both of them.

In dataset HPC, loghub-2.0 labels:
`PSU status ( on off )` -> `PSU status ( <*> <*> )`
`Fan speeds ( 3552 3552 3391 4245 3515 3497 )` -> `Fan speeds ( <*> )`
It raise a question about whether to merge continous placeholders.
also some uncorrected labels happened:
`ambient=40 threshold exceeded` -> `ambient=<*>`

In dataset Hadoop
`TaskAttempt: [attempt_1445087491445_0005_m_000009_0] using containerId: [container_1445087491445_0005_01_000009] on NM: [04DN8IQ.fareast.corp.microsoft.com:55452]`
-> `TaskAttempt: [<*>] using containerId: [<*>] on NM: [<*>]`
`Failed to renew lease for [DFSClient_NONMAPREDUCE_483047941_1] for 46 seconds. Will retry shortly ...`
-> `Failed to renew lease for <*> for <*> seconds. Will retry shortly ...`
It raises a question about whether to include brackets in placeholders.
```

## (b)
<div style="text-align: center; display: flex; justify-content: space-between;">

|  **datasets** | **#Log entries** | **GA** | **MLA** | **ED** | **FGA** | **FTA** |
|:-------------:|-----------------|:------:|:-------:|:------:|:-------:|:-------:|
|    **HPC**    |     429,987     |  0.992 |   0.99  |  0.997 |  0.888  |  0.838  |
| **Proxifier** |      21,320     |    1.000   |    1.000    |    1.000   |    1.000    |    1.000    |
|   **Spark**   |    16,075,117   |  0.974 |  0.997  |  0.999 |  0.832  |  0.702  |
|  **OpenSSH**  |     638,946     | 0.9672 |  0.9976 |  0.999 |  0.800  |  0.800  |

</div>

### A5. Effiency of LogBatcher
We estimate the cost of using OpenAI API using:

```
cost = #tokens * USD$0.003 / 1000
```

The carbon footprint is estimated as follows:
```
Carbon Footprint = Tokens x Energy per Token x gCO2e/KWh
```
`gCO2e/KWh`: 240.6 `gCO2e/KWh` for Microsoft Azure US West

`Energy per Token`: ~4 Joules per output token = 0.001 kWh per 1000 tokens

**Total cost on 16 2k-datasets:**

|                      | **DivLog** | **LILAC** | **LogBatcher** |
|:--------------------:|:----------:|:---------:|:--------------:|
|  #Token per dataset  |   469109   |   24020   |      11701     |
|   Total Cost (USD)   |  22.517232 |  1.15296  |    0.561648    |
| Carbon Footprint (g) | 1805.88201 | 92.467392 |   45.0441696   |

**Cost per LLM invocation:**

|                       | **DivLog** | **LILAC** | **LogBatcher** |
|:---------------------:|:----------:|:---------:|:--------------:|
| #Token per invocation |     235    |    309    |       173      |
|    Total Cost (USD)   |  0.010575  |  0.013905 |    0.007785    |
|  Carbon Footprint (g) |  0.848115  |  1.115181 |    0.624357    |

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

## Results

### 1.Effectiveness, Robustness and Effeciency

- Accuracy comparison with the SOTA Log parsers
<p align="center"><img alt='Effectiveness' src="outputs/figures/Effectiveness.png" width="850"></p>

- Robustness comparison with the SOTA Log parsers
<p align="center"><img alt='Robustness' src="outputs/figures/Robustness.png" width="800"></p>

- Efficiency of LLM-based Log parsers
<p align="center"><img src="outputs/figures/Efficiency.png" width="500"></p>

### 2.Ablation Study

We evaluate the importance of each component by removing each of them from the framework
<p align="center"><img src="outputs/figures/Ablation_study.png" width="400"></p>

### 3.Scalability

- Performance with demonstrations
<p align="center"><img src="outputs/figures/supervised.png" width="400"></p>

- Performance on large-scale datasets
<p align="center"><img src="outputs/figures/big_scale.png" width="800"></p>

### 4.Different Settings

- Batch size
<p align="center"><img src="outputs/figures/batch_size.png" width="200"></p>

- Sampling method
<p align="center"><img src="outputs/figures/Sampling_method.png" width="400"></p>

- LLM selection
<p align="center"><img src="outputs/figures/different_LLMs.png" width="350"></p>
