import json
import re
from openai import OpenAI
from together import Together
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.postprocess import post_process, post_process_for_batch_output
from utils.prune import prune_from_cluster
from utils.sample import nearest_k_pairs_from_log
from utils.util import choose, truncate
from utils.sample_byword import extract_variables, matches_template
from utils.cluster import Cluster
from utils.postprocess import correct_single_template
import httpx

class Cluster_Parser:
    
    def __init__(self, theme, config):
        
        self.model = config['model']
        self.batch_num = config['batch_num']
        self.theme = theme
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, cluster, clusters_num, cached_pairs=[], sample_pairs=[], shot = 0):
        logs =cluster.logs
        length = len(cluster.indexs)
        sample_log = logs[0]
        if type(logs) == str:
            logs = [logs]
        new_cluster = None

        # caching
        for cached_pair in cached_pairs:
            for log in cluster.static_logs:
                match_result = matches_template(log, cached_pair)
                if match_result != None:
                    cluster, new_cluster = prune_from_cluster(
                        cached_pair[1], cluster, clusters_num)
                    print(f"cache hit: {match_result}")
                    return '', match_result, cluster, new_cluster

        demonstrations = ''
        can_match = False

        # using labelled data
        if shot > 0:
            nearest_k_pairs = nearest_k_pairs_from_log(
                sample_log, sample_pairs, shot)
            for i in range(shot):
                # demonstrations += f"\nThe template of log message `{nearest_k_pairs[shot - i - 1][0]}` is `{nearest_k_pairs[shot - i - 1][1]}`." 
                demonstrations += f"Log message: `{nearest_k_pairs[shot - i - 1][0]}`\nLog template: `{nearest_k_pairs[shot - i - 1][1].replace('<*>', '{{variable}}')}`\n"



        # prompt format: instruction + (demonstration) + query(logs)
        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. There might be no variables in the log message." + demonstrations +"\nPrint the input log's template delimited by backticks."
        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. There might be no variables in the log message.\nPrint the input log's template delimited by backticks."
        
        # ablation for clustering
        # instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. There might be no variables in the log message.\nPrint the input log's template separated by line break."

        if demonstrations != '':
            query = demonstrations + 'Log message: ' + '\n'.join([f'`{log}`'for log in logs]) + '\nLog template: '
            # query = 'Log message: ' + '\n'.join([f'`{log}`'for log in logs])
        else:
            query = '\n'.join(logs)
    
        # messages
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content":  query}
        ]

        with open(f'outputs/cost/{self.theme}.json', 'a', encoding='utf-8') as file:
            json.dump(messages, file, ensure_ascii=False, indent=4)
            file.write('\n')
        
        # for i in range(3):
        answer = self.chat(messages)


        tmp, template = post_process(answer)
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
        if can_match:
            cluster, new_cluster = prune_from_cluster(
                template, cluster, clusters_num)
        else:
            template = correct_single_template(sample_log)
            print(f"can not match any log in this batch, return a sampled log as template")

        # ablation for clustering:
        # tmp = ''
        # match_template = ''
        # templates = post_process_for_batch_output(answer)
        # for template in templates:
        #     for log in logs:
        #         matches = extract_variables(log, template)
        #         if matches != None:
        #             # refine for the empty variable
        #             parts = template.split('<*>')
        #             template = parts[0]
        #             for index, match in enumerate(matches):
        #                 if match != '':
        #                     template += '<*>'
        #                 template += parts[index + 1]
        #             match_template =template
        #             break

        # # pruning
        # if match_template != '':
        #     template = match_template
        #     cluster, new_cluster = prune_from_cluster(
        #         match_template, cluster, clusters_num)
        # else:
        #     template = correct_single_template(sample_log)
        #     print(f"can not match any log in this batch, return a sampled log as template")

        
        print(f"final template: {template}")
        return tmp, template, cluster, new_cluster
