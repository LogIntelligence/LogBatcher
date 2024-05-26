# for a in 4 5 6
# do
#     D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --shot $a
# done
model=(codellama/CodeLlama-7b-Instruct-hf codellama/CodeLlama-70b-Instruct-hf Qwen/Qwen1.5-7B-Chat Qwen/Qwen1.5-72B-Chat mistralai/Mixtral-8x7B-Instruct-v0.1 mistralai/Mistral-7B-Instruct-v0.3)

for i in "${model[@]}"; do
    D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --model "$i"
done
