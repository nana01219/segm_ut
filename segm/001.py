"""ft = 24
pre_epoch = 24

print((pre_epoch>0) & (pre_epoch>=ft))"""

import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand([2,3,4,4])
print(x)

y = 1-x

x = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
print(x.shape)
print(x)

for i in range(10000):
    z = F.gumbel_softmax(x, hard=False, tau=0.01, dim=-1)
    print(x[0,0,0])
    print(z[0,0,0])
    print("----------")
    input()
    z = z[:,:,:,:,0]
    # print(x.shape)
    # print(x)
    if i == 0:
        ret = z.unsqueeze(-1)
    else:
        ret = torch.cat((ret, z.unsqueeze(-1)), dim=-1)

print(ret.mean(dim=-1))

