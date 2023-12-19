import httpx
import yaml
import json
from openai import OpenAI
import openai
import time

class Chat:
    def __init__(self, dataset):
        config = yaml.load(open("llm\config.yaml", "r"), Loader=yaml.FullLoader)
        self.api_keys = config['openai']['api_key']
        self.api_key_index = 0
        self.instruction = config['instructions'][dataset]
        with open(f'dataset\{dataset}\demonstrations.json') as file:
            self.demonstrations = json.load(file)
        self.client = OpenAI(
            base_url="https://oneapi.xty.app/v1",
            api_key=self.api_keys[self.api_key_index],
            http_client=httpx.Client(
                proxies="http://127.0.0.1:7890"
            ),
        )

    def change_api_key(self):
        if self.api_key_index >= len(self.api_keys):
            print("No more api key available")
            return False
        self.api_key_index += 1
        self.client = OpenAI(
            base_url="https://oneapi.xty.app/v1",
            api_key=self.api_keys[self.api_key_index],
            http_client=httpx.Client(
                proxies="http://127.0.0.1:7890"
            ),
        )
        return True

    def reverse_test(self, log_message, shot = 0, VERBOSE=True, only_content = True):
        messages = [{"role": "system", "content": self.instruction}]
        if self.demonstrations is not None:
            for demonstration in self.demonstrations[:2*shot]:
                messages.append(demonstration)
         
        messages.append({"role": "user", "content": "log message:\n" + log_message})

        if VERBOSE:
            print('-'*20)
            print(messages)
            print('-'*20)

        response = []

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )

        except Exception as e:
            if e.code == "insufficient_quota":
                    return "IQ"
            elif e.code == "rate_limit_exceeded":
                if "Limit 200, Used 200, Requested 1" in e.message:
                    return "RL2"
                else:
                    return "RL1"
            else:
                return e

        return response
