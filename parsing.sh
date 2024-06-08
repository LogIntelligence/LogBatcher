# Description: Script to run the evaluation.py script with different parameters
for i in 1 2 3 4 5 6
do
    python evaluation_2k.py --shot $i
done

for i in 5 15 20
do
    python evaluation_2k.py --batch_size $i
done