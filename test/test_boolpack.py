import torch
from lomem import packing


if __name__ == '__main__':
    device='cuda:0'

    is_pass = True

    shape = (4097,)
    x = torch.rand(shape, device=device) > 0.5
    packed_flag = packing.pack_bool_to_uint8(x)
    x_restore = packing.unpack_uint8_to_bool(packed_flag, shape)
    if (x != x_restore).any():
        is_pass = False

    shape = (4097, 32)
    x = torch.rand(shape, device=device) > 0.5
    packed_flag = packing.pack_bool_to_uint8(x)
    x_restore = packing.unpack_uint8_to_bool(packed_flag, shape)
    if (x != x_restore).any():
        is_pass = False

    shape = (4097, 2, 3)
    x = torch.rand(shape, device=device) > 0.5
    packed_flag = packing.pack_bool_to_uint8(x)
    x_restore = packing.unpack_uint8_to_bool(packed_flag, shape)
    if (x != x_restore).any():
        is_pass = False


    shape = (17,)
    x = torch.rand(shape, device=device) > 0.5
    packed_flag = packing.pack_bool_to_uint8(x)
    x_restore = packing.unpack_uint8_to_bool(packed_flag, shape)
    if (x != x_restore).any():
        is_pass = False

    if is_pass:
        print("pass!")
    else:
        print("fail!")