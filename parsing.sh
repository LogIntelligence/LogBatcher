# Description: Script to run the evaluation.py script with different parameters
for i in 2000
do
    python evaluation_exp1.py --chunk_size $i
done