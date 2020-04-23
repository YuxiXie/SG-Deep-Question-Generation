import torch.nn as nn


class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, ipt):
        """
        input:
        reduce_size:
        """
        input_size = list(ipt.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        ipt = ipt.view(*output_size)
        ipt, _ = ipt.max(last_dim, keepdim=True)
        output = ipt.squeeze(last_dim)

        return output