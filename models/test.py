import requests

import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
headers = {"Authorization": "Bearer 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"}


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

prompt = """Log message reverse: given a log message, you can generate the corresponding code that generated the log message. If a log message is given, you will only return the code without additional explainations.
Log message: 
2016-09-28 04:30:31, Info                  CSI    WcpInitialize (wcp.dll version 0.0.0.6) called (stack @0x7fed806eb5d @0x7fefa1c8728 @0x7fefa1c8856 @0xff83e474 @0xff83d7de @0xff83db2f)
Code:
logprintf("WcpInitialize (wcp.dll version %s) called (stack @%x @%X @%x @%x @%x @%x)")
Log message:
2016-09-29 02:04:23, Info                  CBS    Session: 30546354_3192394775 initialized by client SPP.
Code:
"""

output = query({
	"inputs": prompt,
})

print(output)
