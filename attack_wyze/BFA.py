import random
import torch
import operator
import torch.nn as nn
from attack_wyze.data_conversion import *

class BFA(object):
    def __init__(self, criterion, model, k_top=10):
        self.criterion = criterion
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        
        # Identify all target layers (nn.Conv2d and nn.Linear)
        self.module_list = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.module_list.append(name)
                # Ensure the layer has the necessary BFA attributes (8-bit quantization)
                if not hasattr(m, 'N_bits'):
                    m.N_bits = 8
                if not hasattr(m, 'b_w'):
                    # bit-wise basis tensor. MSB is negative for 2's complement.
                    b_w = 2**torch.arange(start=m.N_bits - 1, end=-1, step=-1).unsqueeze(-1).float()
                    b_w[0] = -b_w[0]
                    m.register_buffer('b_w', b_w)

    def flip_bit(self, m):
        if self.k_top is None:
            k_top = m.weight.numel()
        else:
            k_top = min(self.k_top, m.weight.numel())

        # 1. Gradient ranking
        if m.weight.grad is None:
             raise ValueError("Weight grad is None. Make sure you called loss.backward()")
        
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top)
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. Bit gradient
        # b_w is [N_bits, 1]
        # w_grad_topk is [k_top]
        b_grad_topk = w_grad_topk * m.b_w

        # 3. Gradient mask
        b_grad_topk_sign = (b_grad_topk.sign() + 1) * 0.5
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).to(torch.int16)
        w_bin_topk = w_bin[w_idx_topk]

        # b_bin_topk: [N_bits, k_top]
        b_bin_topk = (w_bin_topk.repeat(m.N_bits, 1) & m.b_w.abs().repeat(1, k_top).to(torch.int16)) // m.b_w.abs().repeat(1, k_top).to(torch.int16)
        grad_mask = b_bin_topk ^ b_grad_topk_sign.to(torch.int16)

        b_grad_topk *= grad_mask.float()

        # 4. Global search for max bit gradient
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        
        # 5. Flip bits
        w_bin_topk_flipped = (bit2flip.to(torch.int16) * m.b_w.abs().to(torch.int16)).sum(0, dtype=torch.int16) ^ w_bin_topk
        w_bin[w_idx_topk] = w_bin_topk_flipped
        param_flipped = bin2int(w_bin, m.N_bits).view(m.weight.data.size()).float()

        return param_flipped

    def progressive_bit_search(self, model, data, target):
        model.eval()
   
        output = model(data)
        self.loss = self.criterion(output, target)
        
        # Zero gradients
        for name in self.module_list:
            m = dict(model.named_modules())[name]
            if m.weight.grad is not None:
                m.weight.grad.zero_()

        self.loss.backward()
        self.loss_max = self.loss.item()

        # Search
        while self.loss_max <= self.loss.item():
            self.n_bits2flip += 1
            for name in self.module_list:
                module = dict(model.named_modules())[name]
                clean_weight = module.weight.data.detach().clone()
                attack_weight = self.flip_bit(module)
                module.weight.data = attack_weight
                output = model(data)
                self.loss_dict[name] = self.criterion(output, target).item()
                module.weight.data = clean_weight
     
            max_loss_module = max(self.loss_dict.items(), key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # Apply the best flip
        attack_log = []
        for name in self.module_list:
            if name == max_loss_module:
                module = dict(model.named_modules())[name]
                attack_weight = self.flip_bit(module)
                
                weight_mismatch = attack_weight - module.weight.detach()
                attack_weight_idx = torch.nonzero(weight_mismatch, as_tuple=False)
                
                for i in range(attack_weight_idx.size()[0]):
                    idx_tuple = tuple(attack_weight_idx[i].cpu().numpy())
                    weight_prior = module.weight.detach()[idx_tuple].item()
                    weight_post = attack_weight[idx_tuple].item()
                    
                    tmp_list = [0, # Placeholder for module index
                                self.bit_counter + (i+1),
                                max_loss_module,
                                idx_tuple,
                                weight_prior,
                                weight_post]
                    attack_log.append(tmp_list)

                module.weight.data = attack_weight

        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return attack_log
