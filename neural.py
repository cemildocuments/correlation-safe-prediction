"""Fixed neural architecture; optional Metal GPU acceleration, CPU fallback."""
import numpy as np
import torch
from torch import nn

class NeuralClassifier:
    def __init__(self,seed=0,epochs=60):self.seed=seed;self.epochs=epochs
    def fit(self,X,y):
        torch.manual_seed(self.seed);np.random.seed(self.seed);torch.set_num_threads(2)
        self.classes_=np.unique(y);self.device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.net=nn.Sequential(nn.Linear(X.shape[1],128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,len(self.classes_))).to(self.device)
        xx=torch.as_tensor(X,dtype=torch.float32,device=self.device);yy=torch.as_tensor(y,dtype=torch.long,device=self.device)
        opt=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.0001)
        gen=torch.Generator().manual_seed(self.seed)
        self.net.train()
        for epoch in range(self.epochs):
            order=torch.randperm(len(y),generator=gen)
            for ix in order.split(1024):
                ix=ix.to(self.device);opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(self.net(xx[ix]),yy[ix]);loss.backward();opt.step()
        self.net.eval()
        if self.device.type=='mps':torch.mps.synchronize()
        return self
    def predict_proba(self,X):
        ans=[]
        with torch.no_grad():
            for a in np.array_split(X,max(1,int(np.ceil(len(X)/4096)))):
                ans.append(torch.softmax(self.net(torch.as_tensor(a,dtype=torch.float32,device=self.device)),1).cpu().numpy())
        return np.concatenate(ans)
