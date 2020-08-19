import subprocess
print('===========')
cmd = ['docker-compose', 'up', '-d']
print('===========')
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('===========')
out, error = p.communicate()
print(out)
print('===========')
print(error)
print('===========')
print('DONE!')