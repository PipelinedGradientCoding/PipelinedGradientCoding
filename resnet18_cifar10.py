import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data.dataset import random_split
import torch.optim as optim
import numpy as np
import os
import sys

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class GC_FR():
    def __init__(self, config):
        assert n % (s + 1) == 0
        self.config = config
        self.n = self.config["n"]
        self.s = self.config["s"]
        self.c = self.config["c"]

    def encode(self, G, rank):
        return sum(G)

    def decode(self, G, rank):
        y = G[0]
        s = [0 for i in range(self.n // (self.s + 1))]
        s[rank[0] // (self.s + 1)] = 1
        for i in range(1, len(rank)):
            if s[rank[i] // (self.s + 1)] == 0:
                y = y + G[i]
                s[rank[i] // (self.s + 1)] = 1
        return y

    def partition(self, partitions, rank):
        return [partitions[i] for i in range(((rank) // (self. s + 1)) * (self.s + 1), ((rank) // (self. s + 1) + 1) * (self.s + 1))]

    def correct(self, g, rank):
        return g

class GC_CR():
    def __init__(self, config):
        self.config = config        
        self.n = self.config["n"]
        self.s = self.config["s"]

        if self.s != 0:
            np.random.seed(self.config["seed"])
            a = np.random.rand(self.n - 1, self.n - 1)
            q, _ = np.linalg.qr(a)
            self.H = np.zeros((self.s, self.n))
            self.H[0:self.s,0:(self.n-1)] = q[0:self.s]
            self.H[:, -1] = -np.sum(self.H[:, 0:-1], 1)
            self.B = np.zeros((self.n, self.n))

            for i in range(self.n):
                j = np.remainder(np.arange(i, self.s+i+1), self.n)
                self.B[i,j] = np.append([1], -np.linalg.lstsq(self.H[:, j[1: s+1]], self.H[:, j[0]], rcond=None)[0])
        
    def encode(self, G, rank):
        if self.s != 0:
            x = self.B[rank][np.remainder(np.arange(rank, rank + self.s + 1), self.n)]
            y = G[0] * x[0]
            for i in range(1, len(x)):
                y += G[i] * x[i]
            return y
        else:
            return G[0]
    
    def decode(self, G, ranks):
        if self.s != 0:
            ones = np.ones((1, self.n))[0]
            x = np.linalg.lstsq(self.B[ranks].T, ones, rcond=None)
            y = G[0] * x[0][0]
            for i in range(1, len(x[0])):
                y += G[i] * x[0][i]
            return y
        else:
            return sum(G)

    def partition(self, partitions, rank):
        return [partitions[i] for i in np.remainder(np.arange(rank, rank + self.s + 1), self.n)]

    def correct(self, g, rank):
        return g

class IS_GC_FR():
    def __init__(self, config):
        self.config = config
        self.n = self.config["n"]
        self.s = self.config["s"]
        self.c = self.config["c"]

    def encode(self, G, rank):
        return sum(G)

    def decode(self, G, rank, start=0):
        y = G[0]
        count = 1
        s = [0 for i in range(self.n // self.c)]
        s[rank[0] // self.c] = 1
        for i in range(1, len(rank)):
            if s[rank[i] // self.c] == 0:
                y = y + G[i]
                count += 1
                s[rank[i] // self.c] = 1
        return y, count * self.c

    def partition(self, partitions, rank):
        return [partitions[i] for i in range(((rank) // self.c) * self.c, ((rank) // self.c + 1) * self.c)]

class IS_GC_CR(IS_GC_FR):
    def partition(self, partitions, rank):
        return GC_CR.partition(self, partitions, rank)

    def decode(self, G, ranks, start=0):
        def index_max_d(L, c, n):
            if len(L) == 1:
                return L
            L = sorted(L)
            origin = start
            curr = origin
            result = []
            while (L[curr] - L[origin]) % n < c:
                l = [L[curr]]
                head = curr
                nex = (head + 1) % len(L)
                while True:
                    if (L[curr] - L[nex]) % n < c or nex == curr:
                        break
                    if (L[nex] - L[head]) % n >= c:
                        l.append(L[nex])
                        head = nex
                    nex = (nex + 1) % len(L)
                if len(l) > len(result):
                    result = l
                curr = (curr + 1) % len(L)
                if curr == origin:
                    break
            return result

        G0 = {}
        for i in range(len(ranks)):
            G0[ranks[i]] = G[i]
        useful_workers = index_max_d(ranks, self.c, self.n)
        return sum([G0[i] for i in useful_workers]), len(useful_workers) * self.c

class Pipelined_GC_CR(GC_CR):
    def __init__(self, config):
        self.config = config
        self.n = self.config["n"]
        self.s = self.config["s"]
        self.dev = 0

        if self.s != 0:
            np.random.seed(self.config["seed"])
            self.B = np.zeros((self.n, self.n))
            for i in range(self.n):
                self.B[i, i] = 1
                for j in range(1, self.s + 1):
                    self.B[i, (i + j) % self.n] = np.random.uniform(1 - self.dev, 1 + self.dev)

    def decode(self, G, ranks):
        if self.s != 0:
            mu = self.n / (self.n - self.s) / (self.s + 1)
            y = G[0] * mu
            for i in range(1, len(G)):
                y += G[i] * mu
            return y
        else:
            return sum(G)
   
from torchvision import models
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet = models.resnet18(weights="DEFAULT")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)

def train_(batch_idx, model, device, train_loader, optimizer, rank):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    (data, target) = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct

def test(model, device, test_loader):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(
                dim=1,
                keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return {
        "loss": test_loss,
        "accuracy": 100. * correct / len(test_loader.dataset)
    }

def dataset_creator(rank, coder, use_cuda, config):
    n_workers = coder.n
    s_workers = coder.s
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    from filelock import FileLock
    import torchvision.transforms as tt
    from torchvision.datasets import cifar

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose([tt.ToTensor(),
                             tt.Normalize(*stats,inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    with FileLock("./data.lock"):
        train_dataset=cifar.CIFAR10('./cifar10',train=True,transform=train_tfms,download=True)
    random_seed = config["seed"]
    L = [int(len(train_dataset) / n_workers)] * n_workers
    L[-1] += len(train_dataset) % n_workers
    partitions = random_split(train_dataset, L, generator=torch.Generator().manual_seed(random_seed))

    batch_test = config["test_batch"]
    if rank == 0:
        test_dataset=cifar.CIFAR10('./cifar10',train=False,transform=valid_tfms)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_test, shuffle=False)
    else:
        test_loader = None

    return coder.partition(partitions, rank), test_loader

class Network(object):
    def __init__(self, config, rank):
        self.config = config
        torch.set_default_dtype(torch.float64)
        seed_torch(self.config["seed"])
        self.rank = rank 
        self.n_workers = self.config["n"] 
        self.s_workers = self.config["s"]
        self.coder = None
        if self.config["coder"] == "FR":
            self.coder = GC_FR(self.config)
        if self.config["coder"] == "CR":
            if config["pipeline"] == True:
                self.coder = Pipelined_GC_CR(self.config)
            else:
                self.coder = GC_CR(self.config)
        if self.config["coder"] == "IS-FR":
            self.coder = IS_GC_FR(self.config)
        if self.config["coder"] == "IS-CR":
            self.coder = IS_GC_CR(self.config)
        self.batch_train = self.config["train_batch"]
        self.batch_idx = 0
        self.use_cuda = torch.cuda.is_available()
        # self.device = device = torch.device("cuda" if self.use_cuda else "cpu")
        self.device = device = "cpu"
        self.train_datasets, self.test_loader = dataset_creator(self.rank, self.coder, self.use_cuda, self.config)
        kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}
        self.train_loaders = [torch.utils.data.DataLoader(self.train_datasets[i], batch_size=int(self.batch_train / self.n_workers), shuffle=False, generator=torch.Generator().manual_seed(self.config["seed"]), **kwargs) for i in range(len(self.train_datasets))]

        self.models = []
        self.optimizers = []
        for i in range(len(self.train_loaders)):
            self.models.append(Model().to(device))
            self.optimizers.append(optim.SGD(self.models[-1].parameters(), lr=self.config["lr"], momentum=self.config["momentum"]))

        self.loss_correct = [None for i in range(len(self.train_loaders))]

    def update_batch_idx(self):
        self.batch_idx = (self.batch_idx + 1) % self.get_batches() 
        if self.batch_idx == 0:
            kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}
            self.train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=int(self.batch_train / self.n_workers), shuffle=False, generator=torch.Generator().manual_seed(self.config["seed"]+self.batch_idx), **kwargs) for dataset in self.train_datasets]

    def train(self):
        for i in range(len(self.train_loaders)):
            loss, corr = train_(self.batch_idx, self.models[i], self.device, self.train_loaders[i], self.optimizers[i], (self.rank, i))
            self.loss_correct[i] = np.array([loss.detach().numpy(), corr])
        self.update_batch_idx()

    def test(self):
        return test(self.models[0], self.device, self.test_loader)

    def get_gradients(self):
        return {'rank' : self.rank,
                'grad' : [self.coder.encode([self.optimizers[j].param_groups[0]['params'][i].grad for j in range(len(self.optimizers))], self.rank) for i in range(len(self.optimizers[0].param_groups[0]['params']))],
                'loss_correct': self.coder.encode(self.loss_correct, self.rank)
        }

    def set_gradients(self, grads):
        for i in range(len(grads)):
            for j in range(len(self.optimizers)):
                self.optimizers[j].param_groups[0]['params'][i].grad = grads[i].type(torch.DoubleTensor)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
            
    def save(self):
        torch.save(self.models[0].state_dict(), "./resnet18_cifar10_initialized.pt")

    def load(self):
        for model in self.models:
            model.load_state_dict(torch.load("./resnet18_cifar10_initialized.pt"))

    def get_batches(self):
        return len(self.train_loaders[0])

    def get_samples(self):
        return len(self.train_datasets[0])

    def test_(self):
        return [opt.param_groups[0]['params'][-1].grad for opt in self.optimizers]

    def pre_train(self):
        pass

class PipelinedNetwork(Network):
    def __init__(self, config, rank):
        super(PipelinedNetwork, self).__init__(config, rank)
        self.current = 0
        self.current_static = 0
        self.step = config["step"]

    def pre_train(self):
        for _ in range(len(self.optimizers)):
            self.train()

    def train(self):
        for j in range(self.step):
            i = self.current
            loss, corr = train_(self.batch_idx, self.models[i], self.device, self.train_loaders[i], self.optimizers[i], (self.rank, i))
            self.loss_correct[i] = np.array([loss.detach().numpy(), corr])
            self.update_batch_idx()
            self.current_static = self.current
            self.current = (self.current + 1) % len(self.models)

    def get_gradients(self):
        return {'rank' : self.rank,
                'grad' : [self.coder.encode([self.optimizers[i].param_groups[0]['params'][j].grad for i in range(len(self.optimizers))], self.rank) for j in range(len(self.optimizers[0].param_groups[0]['params']))],
                'loss_correct': self.loss_correct[self.current_static]
        }

    def set_gradients(self, grads):
        self.models[self.current].load_state_dict(self.models[self.current_static].state_dict())
        for i in range(len(grads)):
            self.optimizers[self.current].param_groups[0]['params'][i].grad = grads[i].type(torch.DoubleTensor)

    def step(self):
        self.optimizers[self.current].step()

    def correct(self, rank):
        pass

# @ray.remote
def train(config):
    time_steps_accumulated = 0
    n_epochs = config["epoch"]
    is_s = config["is_s"]
    if is_s == 0:
        num_return = n - s
        num_recover = n
    else:
        num_return = n - is_s
        num_recover = n - is_s        
    
    log_interval = 1
    time_interval = 0
    c = 0
    repeat = 0

    current_loss = sys.float_info.max
    for i in range(n_epochs):
        for j in range(int(batches)):
            time_step_st = time.time()
            [actor.train.remote() for actor in NetworkActors]
            grads = [actor.get_gradients.remote() for actor in NetworkActors]
            ready, remaining = ray.wait(grads, num_returns = num_return)
            grads = [ray.get(ref) for ref in ready]
            ranks = [grad['rank'] for grad in grads]
            loss_correct = [grad['loss_correct'] for grad in grads]
            grads = [grad['grad'] for grad in grads]

            if config["coder"] == "IS-FR" or config["coder"] == "IS-CR":
                decoded_gradients = [coder.decode([grads[i][k] for i in range(len(grads))], ranks) for k in range(len(grads[0]))]
                num_recover = decoded_gradients[0][-1]
                averaged_gradients = [(g / num_recover).type(torch.FloatTensor) for g, w in decoded_gradients]
                decoded_loss_correct = coder.decode(loss_correct, ranks)[0][0] / num_recover
                current_loss = decoded_loss_correct
            else:
                averaged_gradients = [coder.decode([grads[i][k] for i in range(len(grads))], ranks) for k in range(len(grads[0]))]
                averaged_gradients = [(g / num_recover).type(torch.FloatTensor) for g in averaged_gradients]
                if config["pipeline"] is True:
                    decoded_loss_correct = np.mean([loss[0] for loss in loss_correct])
                else:
                    decoded_loss_correct = coder.decode(loss_correct, ranks)[0] / num_recover
                current_loss = decoded_loss_correct

            grads_id = ray.put(averaged_gradients)
            [actor.set_gradients.remote(grads_id) for actor in NetworkActors]
            [actor.step.remote() for actor in NetworkActors]
            
            time_step_ed = time.time()
            time_step = time_step_ed - time_step_st
            time_steps_accumulated += time_step
            time_interval += time_step
            if j % log_interval == 0:
                print("{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.4f}".format(i, j, i * int(batches) + j, time_interval/log_interval, time_steps_accumulated, current_loss))
                time_interval = 0
                if "threshold" in config and "repeat" in config:
                    if current_loss <= config["threshold"]:
                        repeat += 1
                    else:
                        repeat = 0
                    if "repeat" in config and repeat >= config["repeat"]:
                        return
            c += 1

import argparse

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

if __name__ == '__main__':
    config = {
        "coder": "FR",
        "n": 12,
        "s": 0,
        "c": 2,
        "train_batch": 256,
        "test_batch": 1024,
        "lr": 0.01,
        "momentum": 0,
        "epoch": 20,
        "seed": 1,
        "pipeline": False,
        "step": 1,
        "repeat": 3,
        "is_s": 1,
        "threshold": 0.5
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()
    if args.kwargs is not None:
        for k, v in args.kwargs.items():
            if k == "coder":
                config[k] = v
            elif k == "correct" or k == "pipeline":
                config[k] = ((v == 'True') or (v == 'true'))
            elif k == "lr" or k == "momentum" or k == "threshold":
                config[k] = float(v) 
            else:
                config[k] = int(v)
    print(config)

    n = config["n"]
    s = config["s"]

    import ray
    ray.init()
    if config["pipeline"]:
        RemoteNetwork = ray.remote(PipelinedNetwork)
    else:
        RemoteNetwork = ray.remote(Network)

    NetworkActors = [RemoteNetwork.remote(config, i) for i in range(n)]
    # NetworkActors[0].save.remote()
    for i in range(0, len(NetworkActors)):
        NetworkActors[i].load.remote()
    batches = ray.get(NetworkActors[0].get_batches.remote())
    samples = ray.get(NetworkActors[0].get_samples.remote())
    for actor in NetworkActors:
        actor.pre_train.remote()

    coder = None
    if config["coder"] == "FR":
        coder = GC_FR(config)
    if config["coder"] == "CR":
        if config["pipeline"] == True:
            coder = Pipelined_GC_CR(config)
        else:
            coder = GC_CR(config)
    if config["coder"] == "IS-FR":
        coder = IS_GC_FR(config)
    if config["coder"] == "IS-CR":
        coder = IS_GC_CR(config)

    train(config)
    exit(0)