import openai
import json
import os
import pandas as pd
from tqdm import tqdm
from openai.embeddings_utils import get_embedding
openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # has been disabled. You need to use your own openai API key.

if os.path.exists("embeddings") == False:
    os.mkdir("embeddings")
input_dir = "logs/"
output_dir = "embeddings/"
log_list = ['HDFS', 'Spark', 'BGL', 'Windows', 'Linux', 'Andriod', 'Mac', 'Hadoop', 'HealthApp', 'OpenSSH', 'Thunderbird', 'Proxifier', 'Apache', 'HPC', 'Zookeeper', 'OpenStack']

for logs in log_list:
    embedding = dict()
    print("Embedding " + logs + "...")
    i = pd.read_csv(input_dir + '/' + logs + '/' + logs + "_2k.log_structured.csv")
    contents = i['Content']
    for log in tqdm(contents):
        response = get_embedding(log, engine="text-search-babbage-query-001")
        embedding[log] = response
    o = json.dumps(embedding, separators=(',',':'))
    f = open(output_dir + logs + ".json","w")
    f.write(o)
    f.close()

    
        
    



