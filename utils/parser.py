import json
import re
import time
from openai import OpenAI
from together import Together
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.postprocess import post_process
from utils.sample import nearest_k_pairs_from_log
from utils.matching import extract_variables, matches_template, prune_from_cluster
from utils.postprocess import correct_single_template

class Cluster_Parser:
    
    def __init__(self, theme, config):
        
        self.model = config['model']
        self.theme = theme
        self.time_consumption_llm = 0
        if 'gpt' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,   # api_key
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
            
    
    def get_responce(self, cluster, clusters_num, cached_pairs={}, sample_pairs=[], shot = 0):
        logs =cluster.logs
        length = len(cluster.indexs)
        sample_log = logs[0]
        if type(logs) == str:
            logs = [logs]
        new_cluster = None

        # caching
        for template, value_f in cached_pairs.items():
            for log in cluster.static_logs:
                match_result = matches_template(log, [value_f[0], template])
                if match_result != None:
                    cluster, new_cluster = prune_from_cluster(
                        template, cluster, clusters_num)
                    print(f"cache hit: {match_result}")
                    return match_result, cluster, new_cluster
        demonstrations = ''
        can_match = False

        # using labelled data
        if shot > 0:
            nearest_k_pairs = nearest_k_pairs_from_log(
                sample_log, sample_pairs, shot)
            for i in range(shot):
                demonstrations += f"Log message: `{nearest_k_pairs[shot - i - 1][0]}`\nLog template: `{nearest_k_pairs[shot - i - 1][1].replace('<*>', '{{variable}}')}`\n"

        # prompt format: instruction + (demonstration) + query(logs)
        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. There might be no variables in the log message.\nPrint the input log's template delimited by backticks."

        if demonstrations != '':
            query = demonstrations + 'Log message:\n' + '\n'.join([f'`{log}`'for log in logs]) + '\nLog template: '
            # query = 'Log message: ' + '\n'.join([f'`{log}`'for log in logs])
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

        if template == '':
            can_match = False
        else:
            # matching
            for log in logs:
                matches = extract_variables(log, template)
                if matches != None:
                    # refine for the empty variable
                    parts = template.split('<*>')
                    template = parts[0]
                    for index, match in enumerate(matches):
                        if match != '':
                            template += '<*>'
                        template += parts[index + 1]
                    can_match = True
                    break
        # pruning
        if not can_match:
            template = correct_single_template(sample_log)
            # print(f"can not match any log in this batch, return a sampled log as template")
        cluster, new_cluster = prune_from_cluster(
            template, cluster, clusters_num)

        return template, cluster, new_cluster
