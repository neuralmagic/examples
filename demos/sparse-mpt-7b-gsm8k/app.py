import deepsparse
import gradio as gr
from typing import Tuple, List

deepsparse.cpu.print_hardware_capability()

MODEL_ID = "hf:neuralmagic/mpt-7b-gsm8k-pruned60-quant"

DESCRIPTION = f"""
# MPT Sparse Finetuned on GSM8k with DeepSparse 
![NM Logo](https://files.slack.com/files-pri/T020WGRLR8A-F05TXD28BBK/neuralmagic-logo.png?pub_secret=54e8db19db)
Model ID: {MODEL_ID}

🚀 **Experience the power of LLM mathematical reasoning** through [our MPT sparse finetuned](https://arxiv.org/abs/2310.06927) on the [GSM8K dataset](https://huggingface.co/datasets/gsm8k). 
GSM8K, short for Grade School Math 8K, is a collection of 8.5K high-quality linguistically diverse grade school math word problems, designed to challenge question-answering systems with multi-step reasoning. 
Observe the model's performance in deciphering complex math questions, such as "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" and offering detailed step-by-step solutions.
## Accelerated Inferenced on CPUs 
The MPT model runs purely on CPU courtesy of [sparse software execution by DeepSparse](https://github.com/neuralmagic/deepsparse/tree/main/research/mpt). 
DeepSparse provides accelerated inference by taking advantage of the MPT model's weight sparsity to deliver tokens fast!

![Speedup](https://cdn-uploads.huggingface.co/production/uploads/60466e4b4f40b01b66151416/qMW-Uq8xAawhANTZYB7ZI.png)
"""

MAX_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 200

# Setup the engine
pipe = deepsparse.Pipeline.create(
    task="text-generation",
    model_path=MODEL_ID,
    sequence_length=MAX_MAX_NEW_TOKENS,
    prompt_sequence_length=16,
)

def clear_and_save_textbox(message: str) -> Tuple[str, str]:
    return "", message


def display_input(
    message: str, history: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    history.append((message, ""))
    return history


def delete_prev_fn(history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ""
    return history, message or ""
    
with gr.Blocks() as demo:    
    with gr.Row():
        with gr.Column():
            gr.Markdown(DESCRIPTION)
        with gr.Column():
            gr.Markdown("""### MPT GSM Sparse Finetuned Demo""")
            
            with gr.Group():
                chatbot = gr.Chatbot(label="Chatbot")
                with gr.Row():
                    textbox = gr.Textbox(container=False,placeholder="Type a message...",scale=10,)
                    submit_button = gr.Button("Submit", variant="primary", scale=1, min_width=0)
                    
            with gr.Row():
                retry_button = gr.Button("🔄  Retry", variant="secondary")
                undo_button = gr.Button("↩️ Undo", variant="secondary")
                clear_button = gr.Button("🗑️  Clear", variant="secondary")

            saved_input = gr.State()

            gr.Examples(examples=[
            "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
            "Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she eat in 4 weeks?",
            "Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?",],inputs=[textbox],)
        
            max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=DEFAULT_MAX_NEW_TOKENS,
                    minimum=0,
                    maximum=MAX_MAX_NEW_TOKENS,
                    step=1,
                    interactive=True,
                    info="The maximum numbers of new tokens",)
            temperature = gr.Slider(
                label="Temperature",
                value=0.3,
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                interactive=True,
                info="Higher values produce more diverse outputs",
                            )
            top_p = gr.Slider(
                label="Top-p (nucleus) sampling",
                value=0.40,
                minimum=0.0,
                maximum=1,
                step=0.05,
                interactive=True,
                info="Higher values sample more low-probability tokens",
                            )
            top_k = gr.Slider(
                label="Top-k sampling",
                value=20,
                minimum=1,
                maximum=100,
                step=1,
                interactive=True,
                info="Sample from the top_k most likely tokens",
                )
            repetition_penalty = gr.Slider(
                label="Repetition penalty",
                value=1.2,
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                interactive=True,
                info="Penalize repeated tokens",
                )

            # Generation inference
            def generate(
                        message,
                        history,
                        max_new_tokens: int,
                        temperature: float,
                        top_p: float,
                        top_k: int,
                        repetition_penalty: float,
                ):
                    generation_config = { "max_new_tokens": max_new_tokens,"temperature": temperature,"top_p": top_p,"top_k": top_k,"repetition_penalty": repetition_penalty,}
                    inference = pipe(sequences=message, streaming=True, **generation_config)
                    history[-1][1] += message
                    for token in inference:
                        history[-1][1] += token.generations[0].text
                        yield history
                    print(pipe.timer_manager)
            textbox.submit(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
                ).then(
                        fn=display_input,
                        inputs=[saved_input, chatbot],
                        outputs=chatbot,
                        api_name=False,
                        queue=False,
                ).success(
                    generate,
                    inputs=[
                            saved_input,
                            chatbot,
                            max_new_tokens,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                    ],
                        outputs=[chatbot],
                        api_name=False,
                    )
                        
            submit_button.click(
                            fn=clear_and_save_textbox,
                            inputs=textbox,
                            outputs=[textbox, saved_input],
                            api_name=False,
                            queue=False,
            ).then(
                            fn=display_input,
                            inputs=[saved_input, chatbot],
                            outputs=chatbot,
                            api_name=False,
                            queue=False,
                ).success(
                            generate,
                            inputs=[saved_input, chatbot, max_new_tokens, temperature],
                            outputs=[chatbot],
                            api_name=False,
                )   
                    
            retry_button.click(
                            fn=delete_prev_fn,
                            inputs=chatbot,
                            outputs=[chatbot, saved_input],
                            api_name=False,
                            queue=False,
            ).then(
                            fn=display_input,
                            inputs=[saved_input, chatbot],
                            outputs=chatbot,
                            api_name=False,
                            queue=False,
            ).then(
                            generate,
                            inputs=[saved_input, chatbot, max_new_tokens, temperature],
                            outputs=[chatbot],
                            api_name=False,
                ) 
            undo_button.click(
                            fn=delete_prev_fn,
                            inputs=chatbot,
                            outputs=[chatbot, saved_input],
                            api_name=False,
                            queue=False,
                ).then(
                            fn=lambda x: x,
                            inputs=[saved_input],
                            outputs=textbox,
                            api_name=False,
                            queue=False,
                    )
            clear_button.click(
                            fn=lambda: ([], ""),
                            outputs=[chatbot, saved_input],
                            queue=False,
                            api_name=False,
                    )
                    
                
            
            
demo.queue().launch()
     