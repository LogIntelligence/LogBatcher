# Description: Script to run the evaluation.py script with different parameters

for i in 5 10 20
do
    python evaluation.py --batch_size $i
done