from paramiko import SSHClient
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run_id','-i',type=str,default=None)

args = parser.parse_args()

run_id = args.run_id

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('lambda.cs.uchicago.edu',username='hands')
sftp_client = ssh.open_sftp()
remote_dir = f'/local/hands/PtensBenchmark/example_model/runs'
if run_id is None:
    runs = sftp_client.listdir(remote_dir)
    runs = [int(s) for s in runs]
    run_id = max(runs) if len(runs) > 1 else runs[0]

remote_dir = f"{remote_dir}/{run_id}"
#transport = ssh.get_transport()
file_names = sftp_client.listdir(remote_dir)
if not os.path.exists('./tmp/'):
    os.mkdir('./tmp/')

if not os.path.exists(f'./tmp/{run_id}/'):
    os.mkdir(f'./tmp/{run_id}/')

for file_name in file_names:
    remote_path = f'{remote_dir}/{file_name}'

    local_path = f'./tmp/{run_id}/{file_name}'
    if not os.path.exists(local_path) or os.stat(local_path).st_mtime <= sftp_client.stat(remote_dir).st_mtime: # type: ignore
        print(f"downloading {file_name}")
        sftp_client.get(remote_path,local_path)