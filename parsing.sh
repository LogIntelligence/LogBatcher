# for a in 4 5 6
# do
#     D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --shot $a
# done
model=(mistralai/Mixtral-8x7B-Instruct-v0.1 mistralai/Mistral-7B-Instruct-v0.3)

# for i in "${model[@]}"; do
#     D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --model "$i"
# done

# for a in 5 15 20
# do
#     D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --shot 0 --batch_size $a
# done

# for a in 1 2 4 5 6
# do
#     D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --shot $a
# done

for i in 1 2 3
do
    D:/Develop/anaconda/envs/langchain38/python.exe parsing_for_2k.py --rerun $i
done