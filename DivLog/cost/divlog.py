import json
import os
from concurrent.futures import ThreadPoolExecutor
import httpx
from tqdm import tqdm
from openai import OpenAI

CLIENT = OpenAI(
    api_key='sk-proj-5EkdZfTfjJ1GJim17pgQT3BlbkFJHCMqWAOX7dTSGOcFOjrn',   # api_key
    http_client=httpx.Client(
            proxies="http://127.0.0.1:7890"  # proxies
    ),
)



# @retry(stop=stop_after_attempt(20), wait=wait_random_exponential(min=1, max=60))
def get_response(prompt, dataset):
    file_path = f'map/{dataset}.json'

    with open(file_path, 'r') as f:
        look_up_map = json.load(f)

    if prompt not in look_up_map:
        response = CLIENT.ChatCompletion.create(
            model='gpt-3.5-turbo-0125',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
        )
        output = response.choices[0].message['content'].strip('\n')
        look_up_map[prompt] = output

        with open(file_path, 'w') as f:
            json.dump(look_up_map, f, indent=4)
    else:
        output = look_up_map[prompt]

    return output


if __name__ == '__main__':
    dataset = 'Linux'
    file_path = f'map/{dataset}.json'

    if not os.path.exists(file_path):
        with open(file_path, 'w') as file1:
            json.dump({}, file1)

    prompt_list_path = f'cost_divlog_for_{dataset}.json'
    with open(prompt_list_path, 'r') as f:
        prompt_list = json.load(f)
    with ThreadPoolExecutor(max_workers=16) as executor:
        tasks = zip(prompt_list, [dataset] * len(prompt_list))
        results = list(
            tqdm(executor.map(get_response, tasks), total=len(prompt_list)))

    # Optionally save the results to a file or use them further
    with open(f'results_for_{dataset}.json', 'w') as f:
        json.dump(results, f, indent=4)
