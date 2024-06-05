# Description: Script to run the evaluation.py script with different parameters
for i in 1000
do
    python evaluation2.py --chunk_size $i
done