import os
import json

import torch
from peft import PeftModel, LoraConfig

import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM

BASE_MODEL = None
assert (
    BASE_MODEL
), "Please specify a BASE_MODEL in the script, e.g. 'decapoda-research/llama-7b-hf'"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()
final_weight = base_model.model.layers[-1].self_attn.q_proj.weight
final_weight_old = final_weight.clone()
embed_weight = base_model.model.embed_tokens.weight
embed_weight_old = base_model.model.embed_tokens.weight.clone()
head_weight = base_model.lm_head.weight
head_weight_old = base_model.lm_head.weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "./guanaco",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)


assert torch.allclose(first_weight_old, first_weight)

# merge weights
for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True
    layer.self_attn.k_proj.merge_weights = True
    layer.self_attn.o_proj.merge_weights = True
lora_model.base_model.model.model.embed_tokens.merge_weights = True
lora_model.base_model.model.lm_head.merge_weights = True

lora_model.train(False)

# did we do anything?
first_changed = not torch.allclose(first_weight_old, first_weight)
final_changed = not torch.allclose(final_weight_old, final_weight)
embed_changed = not torch.allclose(embed_weight_old, embed_weight)
head_changed = not torch.allclose(head_weight_old, head_weight)
print(
    f'first_changed: {first_changed}\n'
    f'final_changed: {final_changed}\n'
    f'embed_changed: {embed_changed}\n'
    f'head_changed : {head_changed}\n'
)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="4GB"
)
