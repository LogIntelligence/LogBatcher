import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.postprocess import post_process
from utils.util import choose, truncate
from utils.sample_byword import matches_template
import httpx

class Cluster_Parser:
    
    def __init__(self, config):
        self.api_key = config['api_key']
        self.model = config['model']
        self.batch_num = config['batch_num']
        self.instruction_for_batch_logs = config['instruction_for_batch_logs']
        self.instruction_for_one_log = config['instruction_for_one_log']
        self.additional_incontext = config['additional_incontext']
        if config['transfer_url']:
            self.client = OpenAI(
                base_url=config['transfer_url'],  # 中转url
                api_key=self.api_key,                      # api_key
                http_client=httpx.Client(
                    proxies=config['proxies']  # 代理地址
                ),
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,                      # api_key
                http_client=httpx.Client(
                    proxies=config['proxies']  # 代理地址
                ),
            )

    # @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(20))
    def chat(self, messages, add=0):
        print('='  * 20)
        print(messages)
        print('-'  * 20)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0 + add*0.5,
        )
        if response.model != self.model:
            print(f"model error: {response.model}")
        print(response)
        print('=' * 20)
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, f, cluster, cached_pairs=[]):
        logs =cluster.logs
        length = len(cluster.indexs)

        if type(logs) == str:
            logs = [logs]

        tmps = []
        templates = []

        additional_incontext = ''

        for i in range(0, len(logs), self.batch_num):
            batch_logs = logs[i:i+self.batch_num]

            # if all logs's length is 1, and not contain any digit, return the log itself
            # can't handle log like setLightOn(true)
            # if all(len(re.split(' ', log)) == 1 and not any(char.isdigit() for char in log) for log in batch_logs):
            #     return [],batch_logs[0]
            if len(logs) == 1 and not any(char.isdigit() for char in logs[0]):
                additional_incontext = ' There might be no variables in the log message'

            # cache
            
            for cached_pair in cached_pairs:
                match_result = matches_template(batch_logs[0], cached_pair)
                if match_result != None:
                    f.write(f"---------------------------\n")
                    f.write(f"cluster {cluster.label}: len={length}\n")
                    f.write(f"{cluster.oracle_template} (ground truth)\n")
                    f.write(f"{match_result} (match result)\n")
                    f.write(f"---------------------------\n")
                    return [], match_result 
                    # additional_incontext = f"Based on the previous logs, the template is likely to be: {cached_template.replace('<*>', '{{variable}}')}"
                    # break
            # end

            # prompt format: instruction + (demonstration) + query(logs)
            messages = []

            # instruction
            # if len(batch_logs) == 1:
            #     messages.append({"role": "system", "content": self.instruction_for_one_log})
            # else:
            #     messages.append({"role": "system", "content": self.instruction_for_batch_logs})
            messages.append(
                {"role": "system", "content": "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template." + additional_incontext + "\nPrint the input log's template delimited by backticks."}
            )

            # query
            prompt = truncate(batch_logs, 4096)
            messages.append({"role": "user", "content": prompt})


            for i in range(3):
                answer = self.chat(messages, add=i)

                tmp, template = post_process(response = answer, reference_log = batch_logs[0]) # tmp means the template before post process
                if template != '':
                    tmps.append(tmp)
                    templates.append(template)
                    break
                else:
                    pass
            if template == '':
                print(f"no template found in this batch: {batch_logs}, return the log itself")
                tmps.append(batch_logs[0])
                templates.append(batch_logs[0])

        final_template, freq, freq_tmp = choose(tmps, templates)
        if templates == []:
            print("no template found, should inference again")

        f.write(f"---------------------------\n")
        f.write(f"cluster {cluster.label}: len={length}\n")
        f.write(f"{cluster.oracle_template} (ground truth)\n")
        f.write(f"{final_template} (final_template)\n")
        for key, value in freq_tmp.items():
            f.write(f"{key}: {value}\n")
        for key, value in freq.items():
            f.write(f"{key}: {value}\n")
        f.write(f"---------------------------\n")
        return tmps, final_template
