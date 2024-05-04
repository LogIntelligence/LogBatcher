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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0 + add*0.5,
        )
        if response.model != self.model:
            print(f"model error: {response.model}")
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, f, cluster, cached_pairs=[]):
        label, logs, indexs, ground_truth = cluster.label, cluster.logs, cluster.indexs, cluster.oracle_template
    
        length = len(indexs)

        tmps = []
        templates = []

        for i in range(0, len(logs), self.batch_num):
            batch_logs = logs[i:i+self.batch_num]
            # if all logs's length is 1, and not contain any digit, return the log itself
            # can't handle log like setLightOn(true)
            if all(len(re.split(' ', log)) == 1 and not any(char.isdigit() for char in log) for log in batch_logs):
                return [],batch_logs[0]

            # cache
            additional_incontext = ''
            for cached_pair in cached_pairs:
                match_result = matches_template(batch_logs[0], cached_pair)
                if match_result != None:
                    f.write(f"---------------------------\n")
                    f.write(f"cluster {label}: len={length}\n")
                    f.write(f"{ground_truth} (ground truth)\n")
                    f.write(f"{match_result} (match result)\n")
                    f.write(f"---------------------------\n")
                    return [], match_result 
                    # additional_incontext = f"Based on the previous logs, the template is likely to be: {cached_template.replace('<*>', '{{variable}}')}"
                    # break
            # end

            # prompt format: instruction + (demonstration) + query(logs)
            messages = []

            # instruction
            if len(batch_logs) == 1:
                messages.append({"role": "system", "content": self.instruction_for_one_log})
            else:
                messages.append({"role": "system", "content": self.instruction_for_batch_logs})

            # entropy based sampling
            # messages.append({"role": "user", "content": '2017-07-02 15:46:41.445 ksfetch[32435/0x7fff79824000] [lvl=2] main() ksfetch fetching URL (<NSMutableURLRequest: 0x1005110b0> { URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 }) to folder:/tmp/KSOutOfProcessFetcher.YH2CjY1tnx/download'})
            # messages.append({"role": "assistant", "content": '`{{timestamp}} ksfetch[{{process_and_thread_id}}] [lvl={{log_level}}] main() ksfetch fetching URL (<NSMutableURLRequest: {{request_id}}> { URL: {{request_url}} }) to folder:{{folder_path}}`'})


            # query
            prompt = truncate(batch_logs, 4096)
            messages.append(
                {"role": "user", "content": f"{prompt}\n{additional_incontext}".strip('\n')})


            for i in range(3):
                answer = self.chat(messages, add=i)
                tmp, template = post_process(response = answer, reference_log = batch_logs[0]) # tmp means the template before post process
                if template != '':
                    tmps.append(tmp)
                    templates.append(template)
                    break
                else:
                    pass


        final_template, freq, freq_tmp = choose(tmps, templates)
        if templates == []:
            print("no template found, should inference again")

        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        f.write(f"{ground_truth} (ground truth)\n")
        f.write(f"{final_template} (final_template)\n")
        for key, value in freq_tmp.items():
            f.write(f"{key}: {value}\n")
        for key, value in freq.items():
            f.write(f"{key}: {value}\n")
        f.write(f"---------------------------\n")
        return tmps, final_template
