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


def get_layer_ids(state_dict, layer_extractor=r'model[.]layers[.](?P<layer_id>\d+)[.].+'):
    import re
    layer_id_set = set()
    for n, p in state_dict.items():
        match = re.search(layer_extractor, n)
        if match is not None:
            layer_id = int(match.group('layer_id'))
            layer_id_set.add(layer_id)
    return list(layer_id_set)

def get_layernorm_weight(state_dict, layernorm_weight):
    for n, p in state_dict.items():
        if ("LayerNorm" in n) or ("layer_norm" in n):
            layernorm_weight[n] = copy.deepcopy(p)
    return layernorm_weight

def merge_layernorm(model_path, output_model_path):
    shutil.copytree(model_path, output_model_path, dirs_exist_ok=True)

    for safetensors_file in glob.glob(os.path.join(output_model_path, '*safetensors*')):
        os.remove(safetensors_file)

    new_state_keys = set()
    layernorm_weight = {}

    shard_files = ["pytorch_model.bin"]
    
    config_file = os.path.join(output_model_path, "config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        num_hidden_layers = config["num_hidden_layers"]

    for shard_file in shard_files:
        shard_path = os.path.join(output_model_path, shard_file)
        state_dict = torch.load(shard_path, map_location='cpu')
        
        layernorm_weight = get_layernorm_weight(state_dict, layernorm_weight)
        del state_dict

    for shard_file in shard_files:
        shard_path = os.path.join(output_model_path, shard_file)
        state_dict = torch.load(shard_path, map_location='cpu')

        if True:
            head_norm_weight = state_dict['lm_head.layer_norm.weight']
            head_norm_bias = state_dict['lm_head.layer_norm.bias']
            
            storage_dtype = state_dict['lm_head.decoder.weight'].dtype
            lm_head_weight = state_dict['lm_head.decoder.weight'].to(torch.float64) * head_norm_weight.to(torch.float64)
            state_dict['lm_head.decoder.weight'] = lm_head_weight.to(storage_dtype)
            state_dict['lm_head.bias'] = head_norm_bias.t().to(torch.float64)@lm_head_weight.t() + state_dict['lm_head.bias'].to(torch.float64)
            state_dict['lm_head.bias'] = state_dict['lm_head.bias'].to(storage_dtype)
            print('model.norm.weight', 'lm_head.weight')


        if True:
            head_norm_weight = state_dict[f'roberta.encoder.layer.{num_hidden_layers-1}.output.LayerNorm.weight']
            head_norm_bias = state_dict[f'roberta.encoder.layer.{num_hidden_layers-1}.output.LayerNorm.bias']
            
            storage_dtype = state_dict['lm_head.dense.weight'].dtype
            lm_head_weight = state_dict['lm_head.dense.weight'].to(torch.float64) * head_norm_weight.to(torch.float64)
            state_dict['lm_head.dense.weight'] = lm_head_weight.to(storage_dtype)
            state_dict['lm_head.dense.bias'] = head_norm_bias.t().to(torch.float64)@lm_head_weight.t() + state_dict['lm_head.dense.bias'].to(torch.float64)
            state_dict['lm_head.dense.bias'] = state_dict['lm_head.dense.bias'].to(storage_dtype)
            print('model.norm.weight', 'lm_head.weight')


        for layer_id in range(num_hidden_layers):
            n = f'roberta.encoder.layer.{layer_id}.intermediate.dense.weight'
            input_layernorm_weight = layernorm_weight[f'roberta.encoder.layer.{layer_id}.attention.output.LayerNorm.weight']
            input_layernorm_bias = layernorm_weight[f'roberta.encoder.layer.{layer_id}.attention.output.LayerNorm.bias']
            proj_weight = state_dict[n]
            proj_bias = state_dict[n.replace("weight", "bias")]
            storage_dtype = proj_weight.dtype
            state_dict[n] = proj_weight.to(torch.float64) * input_layernorm_weight.to(torch.float64)
            state_dict[n.replace("weight", "bias")] = input_layernorm_bias.t().to(torch.float64) @ proj_weight.to(torch.float64).t() + proj_bias
            state_dict[n] = state_dict[n].to(storage_dtype)
            state_dict[n.replace("weight", "bias")] = state_dict[n.replace("weight", "bias")].to(storage_dtype)
            print(f'roberta.encoder.layer.{layer_id}.attention.output.LayerNorm.weight', n)
            

        layer_ids = get_layer_ids(state_dict)
        for n, p in state_dict.items():
            for proj in ['query.weight', 'key.weight', 'value.weight']:
                if proj in n:
                    layer_id = int(re.findall(r'\d+', n)[0])
                    if layer_id == 0:
                        # We don't merge embeddings.LayerNorm
                        input_layernorm_name = 'roberta.embeddings.LayerNorm'
                    else:
                        input_layernorm_name = f'roberta.encoder.layer.{layer_id-1}.output.LayerNorm'
                        input_layernorm_weight = layernorm_weight[f'{input_layernorm_name}.weight']
                        input_layernorm_bias = layernorm_weight[f'{input_layernorm_name}.bias']
                        proj_weight = state_dict[n]
                        proj_bias = state_dict[n.replace("weight", "bias")]
                        storage_dtype = proj_weight.dtype
                        state_dict[n] = proj_weight.to(torch.float64) * input_layernorm_weight.to(torch.float64)
                        state_dict[n.replace("weight", "bias")] = input_layernorm_bias.t().to(torch.float64) @ proj_weight.to(torch.float64).t() + proj_bias
                        state_dict[n] = state_dict[n].to(storage_dtype)
                        state_dict[n.replace("weight", "bias")] = state_dict[n.replace("weight", "bias")].to(storage_dtype)
                        print(input_layernorm_name, n)

        new_state_keys.update(state_dict.keys())
        torch.save(state_dict, shard_path)
        del state_dict
        gc.collect()
    del layernorm_weight
    gc.collect()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LayerNorm into model weights")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the input model")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the output model")
    
    args = parser.parse_args()

    merge_layernorm(args.model_path, args.output_model_path)