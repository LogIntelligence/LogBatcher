# Description: Script to run the evaluation.py script with different parameters
for i in 5 15 20
do
    python evaluation_2k.py --batch_size $i
done