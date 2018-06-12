import subprocess

batcmd="locate boost|grep hpp|grep usr|grep local"
result = subprocess.check_output(batcmd, shell=True)
print(result)