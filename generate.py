import sys

import fire
import torch
from peft import PeftModel
import transformers
import gradio as gr

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


PROMPT = '''### Instruction: 
{}
### Input:
{}

### Response:'''

PROMPT_INS = '''### Instruction: 
{}

### Response:'''


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        prompt='',
        temperature=0.5,
        top_p=0.95,
        top_k=45,
        repetition_penalty=1.17,
        max_new_tokens=128,
        **kwargs,
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()
    
    with gr.Blocks() as demo:
        with gr.Row() as row:
            with gr.Column() as col:
                with gr.Row() and gr.Column():
                    answer = gr.TextArea(label="Response")
                    instruction = gr.Textbox(
                        "", label="Instruction", placeholder="Enter instruction", multiline=True
                    )
                    inputs = gr.Textbox(
                        "", label="Input", placeholder="Enter input (can be empty)", multiline=True
                    )
                    run = gr.Button("Run!")
                    
                    def run_instruction(
                        instruction,
                        inputs,
                        temperature=0.5,
                        top_p=0.95,
                        top_k=45,
                        repetition_penalty=1.17,
                        max_new_tokens=128,
                    ):
                        if inputs.strip() == '':
                            now_prompt = PROMPT_INS.format(instruction)
                        else:
                            now_prompt = PROMPT.format(instruction+'\n', inputs)
                        
                        response = evaluate(
                            now_prompt, temperature, top_p, top_k, repetition_penalty, max_new_tokens
                        )
                        return response
                    
                with gr.Row() and gr.Column():
                    temp = gr.components.Slider(minimum=0, maximum=1, value=0.5, label="Temperature")
                    topp = gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                    topk = gr.components.Slider(minimum=0, maximum=100, step=1, value=35, label="Top k")
                    repp = gr.components.Slider(
                        minimum=0, maximum=2, step=0.01, value=1.2, label="Repeat Penalty"
                    )
                    maxt = gr.components.Slider(
                        minimum=1, maximum=2048, step=1, value=512, label="Max tokens"
                    )
                    maxh = gr.components.Slider(
                        minimum=1, maximum=20, step=1, value=5, label="Max history messages"
                    )
                
                run.click(
                    run_instruction, 
                    [instruction, inputs, temp, topp, topk, repp, maxt],
                    answer
                )
            with gr.Column() as col:
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label='Chat Message input')
                clear = gr.Button("Clear")

                def user(user_message, history):
                    for idx, content in enumerate(history):
                        history[idx] = [
                            content[0].replace('<br>', ''),
                            content[1].replace('<br>', '')
                        ]
                    user_message = user_message.replace('<br>', '')
                    return "", history + [[user_message, None]]

                def bot(
                    history,
                    temperature=0.5,
                    top_p=0.95,
                    top_k=45,
                    repetition_penalty=1.17,
                    max_new_tokens=128,
                    maxh=10,
                ):
                    hist = ''
                    for idx, content in enumerate(history):
                        history[idx] = [
                            content[0].replace('<br>', ''),
                            None if content[1] is None else content[1].replace('<br>', '')
                        ]
                    for user, assistant in history[:-1]:
                        user = user
                        assistant = assistant
                        hist += f'User: {user}\nAssistant: {assistant}\n'
                    now_prompt = PROMPT.format(hist, f"User: {history[-1][0]}")
                    print(now_prompt)
                    print()
                    
                    bot_message = evaluate(
                        now_prompt, temperature, top_p, top_k, repetition_penalty, max_new_tokens
                    )
                    history[-1][1] = bot_message
                    
                    history = history[-maxh:]
                    
                    return history

                msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot, [chatbot, temp, topp, topk, repp, maxt, maxh], chatbot
                )
                clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()

if __name__ == "__main__":
    fire.Fire(main)
