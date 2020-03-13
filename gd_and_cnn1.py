import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn import Module
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

"""
配置:
a=1层conv,b个feature,c个class
"""

class FakeData(Dataset):
    def __init__(self, klass_num, dup=100):
        super(FakeData,self).__init__()
        self.klass_num = max(2,klass_num)
        self.data = []
        self.dup = dup
        for _ in range(klass_num):
            self.data.append(torch.randn(3,3))
    
    def __getitem__(self,index):
        label = index// self.dup
        data = self.data[label]
        noise = 0.01*torch.randn(3,3)
        return (data+noise).view(1,3,3), label

    def __len__(self):
        return self.klass_num*self.dup

class OneLayer(Module):
    def __init__(self, feature_num, klass_num):
        super(OneLayer, self).__init__()
        self.feature_num = feature_num
        self.klass_num = klass_num
        self.f = nn.Sequential(
            nn.Conv2d(1,self.feature_num,3,bias=False),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(self.feature_num,self.klass_num)
    
    def forward(self, x):
        x = self.f(x)
        x = x.view(-1,self.feature_num)
        # x = self.classifier(x)
        return x


def run(a,b,c):
    dataset = FakeData(c)
    loader = DataLoader(dataset,20,True,num_workers=1)
    model = OneLayer(b,c)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.5,momentum=0.9,weight_decay=0.001)
    scheduler = StepLR(optimizer,2,0.9)
    celoss = nn.CrossEntropyLoss()


    epochs = 50
    for ep in range(epochs):
        optimizer.zero_grad()
        for data,label in loader:
            score = model(data)
            loss = celoss(score,label)
            loss.backward()
            optimizer.step()
            print(loss)
            if loss==0:
                break
        ## print('---')
        scheduler.step()
    
    print("sample")
    for d in dataset.data:
        print(d)

    print("features")
    print(model.f[0].weight)

    u = dataset.data
    v = model.f[0].weight.detach()
    cos_ = nn.CosineSimilarity(0)
    ll = []
    for i in u:
        lll = []
        for j in range(v.size(0)):
            na = i.view(-1)
            nb = v[j].view(-1)
            lll.append(cos_(na,nb))
        ll.append(lll)
    import pprint
    pprint.pprint(ll)

    



if __name__ == "__main__":
    run(1,3,3)