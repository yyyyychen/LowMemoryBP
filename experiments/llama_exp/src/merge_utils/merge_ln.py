import os
import gc
import glob
import shutil
import json
import torch
import re
import copy
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

def inspect_model_state_dict(model_path, shard_files):
    total_numel = 0
    total_size = 0
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        state_dict = torch.load(shard_path, map_location='cpu')
        for name, param in state_dict.items():
            numel = param.numel()
            total_numel += numel
            element_size = param.element_size()
            total_size += numel * element_size
            print(name, numel, element_size)
        del state_dict
        gc.collect()
    print('total_numel:', total_numel, 'total_size:', total_size)
    return total_numel, total_size

def get_layernorm_weight(state_dict, layernorm_weight):
    for n, p in state_dict.items():
        if "layernorm" in n:
            layernorm_weight[n] = copy.deepcopy(p)
    return layernorm_weight

def merge_layernorm(model_path, output_model_path):
    shutil.copytree(model_path, output_model_path, dirs_exist_ok=True)

    for safetensors_file in glob.glob(os.path.join(output_model_path, '*safetensors*')):
        os.remove(safetensors_file)

    index_file = os.path.join(output_model_path, WEIGHTS_INDEX_NAME)
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)
    
    shard_files = list(set(index["weight_map"].values()))

    new_state_keys = set()
    layernorm_weight = {}

    for shard_file in shard_files:
        shard_path = os.path.join(output_model_path, shard_file)
        state_dict = torch.load(shard_path, map_location='cpu')
        layernorm_weight = get_layernorm_weight(state_dict, layernorm_weight)
        del state_dict

    for shard_file in shard_files:
        shard_path = os.path.join(output_model_path, shard_file)
        state_dict = torch.load(shard_path, map_location='cpu')

        if 'model.norm.weight' in state_dict.keys() and 'lm_head.weight' in state_dict.keys():
            head_norm_weight = state_dict['model.norm.weight']
            storage_dtype = state_dict['lm_head.weight'].dtype
            state_dict['lm_head.weight'] = head_norm_weight.to(torch.float64) * state_dict['lm_head.weight'].to(torch.float64)
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].to(storage_dtype)
            print('model.norm.weight', 'lm_head.weight')

        for n, p in state_dict.items():
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                if proj in n:
                    layer_id = int(re.findall(r'\d+', n)[0])
                    input_layernorm_name = f'model.layers.{layer_id}.input_layernorm.weight'
                    input_layernorm_weight = layernorm_weight[input_layernorm_name]
                    proj_weight = state_dict[n]
                    storage_dtype = proj_weight.dtype
                    state_dict[n] = proj_weight.to(torch.float64) * input_layernorm_weight.to(torch.float64)
                    state_dict[n] = state_dict[n].to(storage_dtype).t()
                    print(input_layernorm_name, n)
        
            for proj in ['up_proj', 'gate_proj']:
                if proj in n:
                    layer_id = int(re.findall(r'\d+', n)[0])
                    post_attention_layernorm_name = f'model.layers.{layer_id}.post_attention_layernorm.weight'
                    post_attention_layernorm_weight = layernorm_weight[post_attention_layernorm_name]
                    proj_weight = state_dict[n]
                    storage_dtype = proj_weight.dtype
                    state_dict[n] = proj_weight.to(torch.float64) * post_attention_layernorm_weight.to(torch.float64)
                    state_dict[n] = state_dict[n].to(storage_dtype).t()
                    print(post_attention_layernorm_name, n)

        new_state_keys.update(state_dict.keys())

        torch.save(state_dict, shard_path)
        del state_dict
        gc.collect()
    del layernorm_weight
    gc.collect()

    total_numel, total_size = inspect_model_state_dict(output_model_path, shard_files)
    index['metadata']['total_size'] = total_size
    index['metadata']['total_numel'] = total_numel

    for k in list(index["weight_map"].keys()):
        if k not in new_state_keys:
            del index["weight_map"][k]

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LayerNorm into model weights")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the input model")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the output model")
    
    args = parser.parse_args()
    
    merge_layernorm(args.model_path, args.output_model_path)