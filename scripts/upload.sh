#!/bin/bash
ip1=ec2-54-197-26-24.compute-1.amazonaws.com
ip2=ec2-52-87-182-53.compute-1.amazonaws.com
ip3=ec2-54-205-32-160.compute-1.amazonaws.com
ip4=ec2-52-87-203-135.compute-1.amazonaws.com

pem_path=~/.ssh/MyFirstKey.pem

ssh -i $pem_path distributed_example.py	ubuntu@$ip1:~/distributed/
ssh -i $pem_path distributed_example.py	ubuntu@$ip2:~/distributed/
ssh -i $pem_path distributed_example.py	ubuntu@$ip3:~/distributed/
ssh -i $pem_path distributed_example.py	ubuntu@$ip4:~/distributed/

ssh -i $pem_path run_on_single_machine.sh	ubuntu@$ip1:~/distributed/
ssh -i $pem_path run_on_single_machine.sh	ubuntu@$ip2:~/distributed/
ssh -i $pem_path run_on_single_machine.sh	ubuntu@$ip3:~/distributed/
ssh -i $pem_path run_on_single_machine.sh	ubuntu@$ip4:~/distributed/
