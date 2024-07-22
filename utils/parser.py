import json
import re
import time
from openai import OpenAI
from together import Together
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.cluster import Cluster
from utils.postprocess import post_process
from utils.sample import nearest_k_pairs_from_log
from utils.matching import extract_variables, matches_template, prune_from_cluster
from utils.postprocess import correct_single_template
import httpx

class Cluster_Parser:
    
    def __init__(self, model, theme, config):
        
        self.model = model
        self.theme = theme
        self.time_consumption_llm = 0
        if 'gpt' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,   # api_key
                http_client=httpx.Client(
                    proxies="http://127.0.0.1:7890"  # proxies
                ),
            )
        else:
            self.api_key = config['api_key_from_together']
            self.client = Together(
                    api_key=self.api_key   # api_key
                )

    # @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(20))
    def chat(self, messages):
        t1 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        self.time_consumption_llm += (time.time() - t1)
        return response.choices[0].message.content.strip('\n')
    
    # @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(20))
    def inference(self, prompt):
        retry_times = 0
        output = ''
        while True:
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.0,
                )
                output = response.choices[0].text.strip('\n')
            except Exception as e:
                print(e)
                retry_times += 1
                if retry_times > 3:
                    return output
            else:
                return output
            
    
    def get_responce(self, cluster, cached_pairs={}, sample_pairs=[], shot = 0):

        # initialize
        logs =cluster.batch_logs
        sample_log = logs[0]
        if type(logs) == str:
            logs = [logs]
        new_cluster = Cluster()
        # caching
        for template, referlog_and_freq in cached_pairs.items():
            for log in cluster.logs:
                match_result = matches_template(log, [referlog_and_freq[0], template])
                if match_result != None:
                    cluster, new_cluster = prune_from_cluster(template, cluster)
                    cached_pairs[template][1] += len(new_cluster.logs)
                    # print(f"cache hit: {match_result}")
                    return match_result, cluster, new_cluster
                

        demonstrations = ''

        # using labelled data
        if shot > 0:
            nearest_k_pairs = nearest_k_pairs_from_log(
                sample_log, sample_pairs, shot)
            for i in range(shot):
                demonstrations += f"Log message: `{nearest_k_pairs[shot - i - 1][0]}`\nLog template: `{nearest_k_pairs[shot - i - 1][1].replace('<*>', '{{variable}}')}`\n"

        # prompt format: instruction + (demonstration) + query(logs)
        instruction = "You will be provided with some log contents separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. There might be no variables in the log content.\nPrint the input log's template delimited by backticks."

        if demonstrations != '':
            query = demonstrations + 'Log message:\n' + '\n'.join([f'`{log}`'for log in logs]) + '\nLog template: '
        elif all(model_tpye not in self.model for model_tpye in ['gpt', 'instruct', 'chat']):
            query = 'Log message:\n' + '\n'.join([f'`{log}`'for log in logs]) + '\nLog template: '
        else:
            query = '\n'.join(logs)
    
        # invoke LLM
        cost_file = open(f'outputs/cost/{self.theme}.json', 'a', encoding='utf-8')
        if any(model_tpye in self.model for model_tpye in ['gpt', 'instruct', 'chat']):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content":  query}
            ]
            json.dump(messages, cost_file, ensure_ascii=False, indent=4)
            cost_file.write('\n')
            answer = self.chat(messages)
        else:
            prompt = f"{instruction}\n{query}"
            json.dump(prompt, cost_file, ensure_ascii=False, indent=4)
            answer = self.inference(prompt)
        cost_file.close()
            

        template = post_process(answer)

        # matching and pruning
        for log in logs:
            matches = extract_variables(log, template)
            if matches != None:
                parts = template.split('<*>')
                template = parts[0]
                for index, match in enumerate(matches):
                    if match != '':
                        template += '<*>'
                    template += parts[index + 1]
                break
        else:
            template = correct_single_template(sample_log)
        cluster, new_cluster = prune_from_cluster(template, cluster)
        return template, cluster, new_cluster
