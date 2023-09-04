from paramiko import SSHClient
import os
from argparse import ArgumentParser


ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('lambda.cs.uchicago.edu',username='hands')
sftp_client = ssh.open_sftp()
remote_dir = f'/local/hands/PtensBenchmark/example_model/ptens.log'

#transport = ssh.get_transport()
local_path = f'./tmp/ptens.log'
sftp_client.get(remote_dir,local_path)