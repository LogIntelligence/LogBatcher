import ast
import csv
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from modeltester import ModelTester
import ast
import tiktoken

def main():
    # get a tester object with data
    openai.api_key = "sk-MWCZbiYqiQUjacuGF53a6c71E3134177A585CeFe79D10aD2"
    openai.proxy = {"http://127.0.0.1:7890", "https://127.0.0.1:7890"}
    print("Parsing " + "Windows" + " ...")

    tester = ModelTester(
        log_path="testDivLog",
        result_path="results",    # .result_csv
        map_path="testDivLog",          # .map_json
        dataset="Windows",       # HDFS, Spark, BGL, Windows, Linux, Andriod, Mac, Hadoop, HealthApp, OpenSSH, Thunderbird, Proxifier, Apache, HPC, Zookeeper, OpenStack
        emb_path="testDivLog",           # embedding
        cand_ratio=0.1,       # ratio of candidate set
        split_method="DPP",   # random or DPP
        order_method="KNN",   # random or KNN
        permutation="ascend",     # permutation
        warmup=False,               # warmup or not
        subname='',             # subname of the files
    )

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Similarist_gt"])  # 写入标题

        for line_idx in tqdm(range(len(tester.log_test[:2000]))):
            if line_idx >= 2000:
                break
            line = tester.log_test[line_idx]
            # get a prompt with five examples for each log message
            prompt, similarist_gt = tester.generatePrompt(line, nearest_num=5)
            writer.writerow([prompt, similarist_gt])
        

    # tester.textModelBatchTest(model='curie',
    #                           model_name='gptC',
    #                           limit=2000,         # number of logs for testing
    #                           N=5,                  # number of examples in the prompt
    #                           )


# df = pd.read_csv("testDivLog\Windows_2k.log_structured.csv")
# print(df)
main()


