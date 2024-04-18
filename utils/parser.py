import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.postprocess import post_process
from utils.util import choose
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

    # @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, f, cluster, cached_templates=[]):
        label, logs, indexs, ground_truth = cluster.label, cluster.logs, cluster.indexs, cluster.oracle_template
    
        length = len(indexs)
        templates = []

        for i in range(0, len(logs), self.batch_num):
            batch_logs = logs[i:i+self.batch_num]
            # if all logs's length is 1, and not contain any digit, return the log itself
            # can't handle log like setLightOn(true)
            if all(len(re.split(' ', log)) == 1 and not any(char.isdigit() for char in log) for log in batch_logs):
                return batch_logs[0]

            # cache
            additional_incontext = ''
            for cached_template in cached_templates:
                if matches_template(batch_logs[0], cached_template):
                    additional_incontext = f"Based on the previous logs, the template is likely to be: {cached_template.replace('<*>', '{{variable}}')}"
                    break
            # end

            messages = []
            if len(batch_logs) == 1:
                messages.append({"role": "system", "content": self.instruction_for_one_log})
            else:
                messages.append({"role": "system", "content": self.instruction_for_batch_logs})

            # add additional incontext
            # if self.additional_incontext:
            #     messages[0]["content"] += self.additional_incontext

            # messages.append({"role": "user", "content": '2017-07-02 15:46:41.445 ksfetch[32435/0x7fff79824000] [lvl=2] main() ksfetch fetching URL (<NSMutableURLRequest: 0x1005110b0> { URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 }) to folder:/tmp/KSOutOfProcessFetcher.YH2CjY1tnx/download'})
            # messages.append({"role": "assistant", "content": '`{{timestamp}} ksfetch[{{process_and_thread_id}}] [lvl={{log_level}}] main() ksfetch fetching URL (<NSMutableURLRequest: {{request_id}}> { URL: {{request_url}} }) to folder:{{folder_path}}`'})
            # batch logs to str
            prompt = ""
            for log in batch_logs:
                prompt += log + '\n'
            # if len(prompt) > 4096:
            #     f.write(f"+++++++++++++++++++++++++++\n")
            #     f.write(f"cluster {label} is out of size, cut it\n")
            #     f.write(f"+++++++++++++++++++++++++++\n")
            #     prompt = ""
            #     for log in batch_logs[:5]:
            #         prompt += log + '\n'
            if additional_incontext:
                messages.append({"role": "assistant", "content": prompt + additional_incontext})
            else:
                messages.append({"role": "user", "content": prompt.strip('\n')})
            answer = self.chat(messages)
            template =  post_process(answer)
            if template != '':
                templates.append(template)

        final_tempalte, freq = choose(templates)

        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        f.write(f"{ground_truth} (ground truth)\n")
        f.write(f"{final_tempalte} (final_template)\n")
        # 打印结果
        for key, value in freq.items():
            f.write(f"{key}: {value}\n")
            # print(f"{key}: {value}")
        f.write(f"---------------------------\n")
        return final_tempalte