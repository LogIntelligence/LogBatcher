import json
from logbatcher.parsing_base import single_dataset_paring
from logbatcher.parser import Parser
from logbatcher.util import data_loader

# load api key, dataset format and parser
model, dataset, folder_name ='gpt-3.5-turbo-0125', 'Apache', 'test'
config = json.load(open('config.json', 'r'))

try:
    parser = Parser(model, folder_name, config)
except Exception as e:
    print(e)
    exit()

# load contents from raw log file, structured log file or content list
contents = data_loader(
    file_name=f"datasets/loghub-2k/{dataset}/{dataset}_2k.log",
    dataset_format= config['datasets_format'][dataset],
    file_format ='raw'
)

# parse logs
single_dataset_paring(
    dataset=dataset,
    contents=contents,
    output_dir= f'outputs/parser/{folder_name}/',
    parser=parser,
    debug=False
)