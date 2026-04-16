import torch

def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    if num_bits == 1:
        output = output/2 + .5
    elif num_bits > 1:
        output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
    return output

def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivalently) back to the signed integer format
    '''
    if num_bits == 1:
        output = input*2-1
    elif num_bits > 1:
        mask = 2**(num_bits - 1) - 1
        output = -(input & ~mask) + (input & mask)
    return output

def weight_conversion(model):
    # This is useful for bit-flipping on layers that follow the AI-Attestation quantization style.
    # For the Wyze model, weights are already in INT8 spirit but stored as floats.
    for m in model.modules():
        if hasattr(m, 'N_bits') and hasattr(m, 'weight'):
            w_bin = int2bin(m.weight.data, m.N_bits).to(torch.int16)
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return
