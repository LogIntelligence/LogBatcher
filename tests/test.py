from logbatcher.parser import Parser
from logbatcher.util import (
    count_message_tokens,
    count_prompt_tokens,
    data_loader,
    generate_logformat_regex,
)
from logbatcher.parsing_base import (
    single_dataset_paring
)

def test_basic():
    dataset = 'Apache'
    config = {
        "api_key_from_openai": "<OpenAI_API_KEY>",
        "api_key_from_together": "<Together_API_KEY>",
        "datasets_format" : {
            "Apache": "\\[<Time>\\] \\[<Level>\\] <Content>",
        }
    }

    try:
        parser = Parser("gpt-3.5-turbo-0125", "test", config)
    except ValueError as e:
        assert str(e) == "Please provide your OpenAI API key and Together API key in the config.json file."

    contents = data_loader(
        file_name=f"datasets/loghub-2k/{dataset}/{dataset}_2k.log",
        dataset_format= config['datasets_format'][dataset],
        file_format ='raw'
    )

    assert len(contents) == 2000