from concurrent.futures import ThreadPoolExecutor
import json
import backoff
import httpx
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from openai import OpenAI
DATASET = 'Linux'
CLIENT = OpenAI(
        api_key='sk-proj-5EkdZfTfjJ1GJim17pgQT3BlbkFJHCMqWAOX7dTSGOcFOjrn',   # api_key
        http_client=httpx.Client(
            proxies="http://127.0.0.1:7890"  # proxies
        ),
    )


@backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=20)
def get_response(prompt):
    f = open(f'map/{DATASET}.json', 'a')
    look_up_map = json.load(f)
    if look_up_map.get(prompt):
        f.close()
        return look_up_map[prompt]
    response = CLIENT.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
    )
    output = response.choices[0].message.content.strip('\n')
    look_up_map[prompt] = output
    return output

if __name__ == '__main__':
    DATASET = 'Linux'
    CLIENT = OpenAI(
            api_key='sk-proj-5EkdZfTfjJ1GJim17pgQT3BlbkFJHCMqWAOX7dTSGOcFOjrn',   # api_key
            http_client=httpx.Client(
                proxies="http://127.0.0.1:7890"  # proxies
            ),
        )
    get_response('hello')

    # dataset = 'Linux'
    # with open(f'cost_divlog_for_{dataset}.json', 'r') as f:
    #     prompt_list = json.load(f)

    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     results = list(tqdm(executor.map(get_response, prompt_list),total=2000))
