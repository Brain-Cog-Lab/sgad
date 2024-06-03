import torch
import torch.nn as nn

# class TransEncoder(nn.Module):
#     def __init__(self, step):
#         super(TransEncoder, self).__init__()
#         self.step = step
#         self.trans = Transformer(dim=128, depth=3, heads=8, dim_head=, mlp_dim, dropout=0.)


class Encoder(nn.Module):
    '''
    (step, batch_size, )
    '''

    def __init__(self, step, device, encode_type='ttfs'):
        super(Encoder, self).__init__()
        self.device = device
        self.step = step
        self.fun = getattr(self, encode_type)
        self.encode_type = encode_type

    def forward(self, inputs, deletion_prob=None, shift_var=None):
        if self.encode_type == 'auto':
            if self.fun.device != inputs.device:
                self.fun.to(inputs.device)

        outputs = self.fun(inputs)
        if deletion_prob:
            outputs = self.delete(outputs, deletion_prob)
        if shift_var:
            outputs = self.shift(outputs, shift_var)
        return outputs

    @torch.no_grad()
    def direct(self, inputs):
        shape = inputs.shape
        outputs = inputs.unsqueeze(0).repeat(self.step, *([1] * len(shape)))
        return outputs

    @torch.no_grad()
    def ttfs(self, inputs):
        # print("ttfs")
        shape = (self.step, ) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        for i in range(self.step):
            mask = (inputs * self.step <=
                    (self.step - i)) & (inputs * self.step >
                                        (self.step - i - 1))
            outputs[i, mask] = 1 / (i + 1)
        return outputs

    @torch.no_grad()
    def rate(self, inputs):
        shape = (self.step, ) + inputs.shape
        return (inputs > torch.rand(shape, device=self.device)).float()
