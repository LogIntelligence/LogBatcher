from concurrent.futures import ThreadPoolExecutor
import httpx
import openai
import json
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
# from openai.embeddings_utils import get_embedding
import argparse


client = OpenAI(
    api_key='sk-zY5LaAEd3EUdBVmKA75aDe77C9684c209b128b981826C043',   # api_key
    base_url='https://api.xty.app/v1',
    http_client=httpx.Client(
        proxies="http://127.0.0.1:7890"  # proxies
    ),
)


if os.path.exists("embeddings") == False:
    os.mkdir("embeddings")
input_dir = "dataset/"
output_dir = "embeddings/"
log_list = ['HDFS', 'Spark', 'BGL', 'Windows', 'Linux', 'Android', 'Mac', 'Hadoop', 'HealthApp', 'OpenSSH', 'Thunderbird', 'Apache', 'HPC', 'Zookeeper', 'OpenStack']

def get_response(log):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=log
    )
    return response.data[0].embedding

for logs in log_list:
    embedding = dict()
    print("Embedding " + logs + "...")
    i = pd.read_csv(input_dir + '/' + logs + '/' + logs + "_2k.log_structured_corrected.csv")
    contents = i['Content']
    with ThreadPoolExecutor(max_workers=16) as executor:
        embeddings = list(
            tqdm(executor.map(get_response, contents),
                 total=len(contents)))

    for log, embed in zip(contents, embeddings):
        embedding[log] = embed

    o = json.dumps(embedding, separators=(',',':'))
    f = open(output_dir + logs + ".json","w")
    f.write(o)
    f.close()
