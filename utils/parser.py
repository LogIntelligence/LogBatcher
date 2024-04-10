import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.postprocess import post_process
from utils.util import choose
import httpx

class Cluster_Parser:
    def __init__(self, config):
        self.api_key = config['api_key']
        self.model = config['model']
        self.batch_num = config['batch_num']
        self.instruction_for_batch_logs = config['instruction_for_batch_logs']
        self.instruction_for_one_log = config['instruction_for_one_log']
        self.addition_incontext = config['additional_incontext']
        if config['transfer_url']:
            self.client = OpenAI(
                base_url=config['transfer_url'],  # 中转url
                api_key=self.api_key,                      # api_key
            )

    # @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, f, cluster):
        label, logs, indexs, ground_truth = cluster.label, cluster.logs, cluster.indexs, cluster.oracle_template
        length = len(indexs)
        templates = []

        for i in range(0, len(logs), self.batch_num):
            batch_logs = logs[i:i+self.batch_num]
            # if all logs's length is 1, and not contain any digit, return the log itself
            if all(len(re.split(' ', log)) == 1 and not any(char.isdigit() for char in log) for log in batch_logs):
                return batch_logs[0]

            messages = []
            if len(batch_logs) == 1:
                messages.append({"role": "system", "content": self.instruction_for_one_log})
            else:
                messages.append({"role": "system", "content": self.instruction_for_batch_logs})


            # batch logs to str
            prompt = ""
            length_prompt = 0
            for log in batch_logs:
                prompt += log + '\n'
            if len(prompt) > 4096:
                f.write(f"+++++++++++++++++++++++++++\n")
                f.write(f"cluster {label} is out of size, cut it\n")
                f.write(f"+++++++++++++++++++++++++++\n")
                prompt = ""
                for log in batch_logs[:5]:
                    prompt += log + '\n'
            messages.append({"role": "user", "content": prompt.strip('\n')})
            answer = self.chat(messages)
            template =  post_process(answer)
            if template != '':
                templates.append(template)

        final_tempalte, freq = choose(templates)

        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        f.write(f"{ground_truth} (ground truth)\n")
        # 打印结果
        for key, value in freq.items():
            f.write(f"{key}: {value}\n")
            # print(f"{key}: {value}")
        f.write(f"---------------------------\n")
        return final_tempalte