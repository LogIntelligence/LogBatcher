import subprocess
for i in [2,3,4,5]:
    subprocess.run(["python", "evaluation_2k.py"] + ["--time", str(i)])
