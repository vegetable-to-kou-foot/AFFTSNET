import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.Tensor([[1, 2, 3]])
b = torch.Tensor([[1, 2, 3]])
c = a*b
print(c)
print(c**2)
b = torch.Tensor([[0.2,0.5],[0.2,0.5]])
target = torch.Tensor([1]).long()
tmp = torch.Tensor([0,1]).long()
tar = F.one_hot(tmp)
s = nn.Softmax(dim=1)
bcel = nn.BCELoss()
logsoftmax = nn.LogSoftmax(dim=1)
ce = nn.CrossEntropyLoss()
nll = nn.NLLLoss()

# 测试CrossEntropyLoss
cel = ce(a, target)
print(cel)
# 输出：tensor(0.4076)

# 测试LogSoftmax+NLLLoss
lsm_a = logsoftmax(a)
nll_lsm_a = nll(lsm_a, target)
print(nll_lsm_a)
# 输出tensor(0.4076)

tmp = s(b)
ans = bcel(tmp,tar.float())
print(ans)

x = torch.Tensor([[1,2,3],[4,5,6]])
y = torch.Tensor([[1,2,3],[4,5,6]])
z = torch.cat((x,y),dim=1)
print(z)

a = torch.zeros([3],dtype=torch.long)
b = F.one_hot(a,num_classes=2)
c = torch.ones([3],dtype=torch.long)
d = F.one_hot(c,num_classes=2)
print(b)
print(d)

a = torch.Tensor([[1,2],[3,4],[6,5]])
print(sum(a[:,0]<a[:,1]).item(),len(a))

