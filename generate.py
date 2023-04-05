import sys

import fire
import torch
from peft import PeftModel
import transformers
import gradio as gr
from utils.callbacks import Iteratorize, Stream
import tqdm

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
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.bfloat16,
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

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

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
        stream_output=False,
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
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield decoded_output.split("### Response:")[-1].strip()
            return  # early return for stream_output
        
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
        yield output.split("### Response:")[-1].strip()
    
    with gr.Blocks() as demo:
        with gr.Row() as row:
            with gr.Column() as col:
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
                    stream_output=False,
                ):
                    if inputs.strip() == '':
                        now_prompt = PROMPT_INS.format(instruction)
                    else:
                        now_prompt = PROMPT.format(instruction+'\n', inputs)
                    
                    response = evaluate(
                        now_prompt, temperature, top_p, top_k, repetition_penalty, max_new_tokens, stream_output
                    )
                    if stream_output:
                        response = tqdm.tqdm(response, unit='token')
                    for i in response:
                        i=i
                        yield i
                
                temp = gr.components.Slider(minimum=0, maximum=1, value=0.5, label="Temperature")
                topp = gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                topk = gr.components.Slider(minimum=0, maximum=100, step=1, value=30, label="Top k")
                repp = gr.components.Slider(
                    minimum=0, maximum=2, step=0.01, value=1.07, label="Repeat Penalty"
                )
                maxt = gr.components.Slider(
                    minimum=1, maximum=2048, step=1, value=512, label="Max tokens"
                )
                maxh = gr.components.Slider(
                    minimum=1, maximum=20, step=1, value=5, label="Max history messages"
                )
                stream_output = gr.components.Checkbox(
                    value=True, label="stream output"
                )
                
                run.click(
                    run_instruction, 
                    [instruction, inputs, temp, topp, topk, repp, maxt, stream_output],
                    answer,
                )
            with gr.Column() as col:
                chat_height = gr.components.Slider(
                    minimum=120, maximum=2000, step=10, value=500, label="Max history messages"
                )
                chat_height.change(
                    None, chat_height, 
                    _js="""(h)=>{chatbot.style.height = h + "px";}"""
                )
                chatbot = gr.Chatbot(label="Guanaco ChatBot", elem_id = "chatbot")
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
                    stream_output=False
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
                    
                    bot_message = evaluate(
                        now_prompt, temperature, top_p, top_k, repetition_penalty, max_new_tokens, stream_output
                    )
                    
                    if stream_output:
                        bot_message = tqdm.tqdm(bot_message, unit='token')
                    for mes in bot_message:
                        mes = mes
                        history[-1][1] = mes
                        
                        history = history[-maxh:]
                        
                        yield history
                
                msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot, [chatbot, temp, topp, topk, repp, maxt, maxh, stream_output], chatbot
                )
                clear.click(lambda: None, None, chatbot, queue=False)
        demo.load(None, chat_height, _js="""
(height)=>{
    chatbot.style.height = height + "px";
    document.querySelector('#chatbot .wrap').style.maxHeight = "calc(100% - 45px)"
}
        """)
    demo.queue().launch()

if __name__ == "__main__":
    fire.Fire(main)
