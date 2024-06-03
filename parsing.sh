# Description: Script to run the evaluation.py script with different parameters
Qwen/Qwen1.5-32B
for i in 5 10 20
do
    python evaluation.py --batch_size $i
done