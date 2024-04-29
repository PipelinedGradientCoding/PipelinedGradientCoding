We have evaluated pipelined gradient coding (PGC) using experiments on the Google Cloud Platform (GCP). Specifically, we implement PGC using Ray for distributed computing and PyTorch for training.  Ray also supports a local simulation when there are enough CPU cores on a single machine. Therefore, we can run the attached program on a local machine to observe the performance. However, when running the program locally, there is no communication cost, and it will affect the performance of some coding schemes, in particular for those codes designed to reduce the communication cost. In this case, however, we can still demonstrate the training performance regarding the number of steps. 

Installing Ray, PyTorch, and related dependencies is required. We list the commands below to install the necessary libraries and dependencies for most cases. In some environments, these commands may not be sufficient. Additional libraries and dependencies may need to be installed. Please follow the system prompts for installation.

```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
pip3 install numpy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install -U "ray[default]"
sudo apt install net-tools
```

When conducting experiments in a cloud environment, it is necessary to set up SSH communication without a password at the very beginning. It is recommended to set up one instance for communication without a password and then duplicate it to construct a distributed computing cluster. Remember to ssh the current machine itself to check whether setting ssh-without-password is successful.

```
cd ~/.ssh 
chmod 700 ~/.ssh 
ssh-keygen -t rsa 
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys 
chmod 600 ~/.ssh/authorized_keys
sudo vim /etc/ssh/ssh_config to add the following:
    StrictHostKeyChecking no 
    UserKnownHostsFile /dev/null
```

When the distributed computing cluster is correctly set up, we can use a general Python 3 launcher to execute our code. In our code, we have packaged all coding schemes mentioned in the paper. We need to pass different parameters to use different coding schemes. We have created a dictionary `config` to contain all variables, allowing insights on correctly using parameters by observing `config`. We provide a sample command below.

```
python3 resnet18_cifar10.py -k coder=CR epoch=2 n=12 s=1 seed=1 is_s=0 pipeline=False 
```

We use `coder` to indicate the coding scheme in the current execution. `epoch` is the number of epochs in training. `n` and `s` are the number of all workers and tolerated stragglers, respectively. `seed` is the random seed in the current execution. To ensure fair comparisons among different coding schemes, we carefully control all random operations, and the models are initialized with the same parameters at the beginning. Therefore, it is recommended to run each scheme with different `seed` to average the performance. `is_s` is used for ignore-straggler SGD, and it has a high priority. When testing other schemes, please ensure that `is_s=0` to avoid any potential errors. `pipeline`, as the name suggests, is used to determine whether to use pipelined training or normal training. In a word, the command above is for gradient coding (GC) with a cyclic repetition (CR) scheme, and it can tolerate one straggler. There might be some other required arguments. For example, if testing ignore-straggler gradient coding (IS-GC), it is necessary to indicate the value of `c` to denote the number of dataset partitions on each worker.

We have also attached an initialized model, and it is necessary to place this model in the same directory. If not, the program would report an error to say that there is no initialized model. In this case, we can generate a new initialized model by hiding the `load()` in the code and running the training for $1$ step only. Remember to use `save()` to save the initialized model. 



