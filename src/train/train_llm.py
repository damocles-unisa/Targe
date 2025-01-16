import warnings
import os

import wandb
import huggingface_hub

from libs.finetuning.train import train

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

wandb.login(key="")
huggingface_hub.login(token="")

def run_experiment(model: str, data_path: str, prompt: dict, num_epochs: int, cutoff_len: int, group_by_length: bool,
                   output_dir: str, lora_target_modules: list, lora_r: int, wandb_project: str, micro_batch_size: int):
    train(base_model=model,
          data_path=data_path,
          prompt=prompt,
          num_epochs=num_epochs,
          cutoff_len=cutoff_len,
          group_by_length=group_by_length,
          output_dir=output_dir,
          lora_target_modules=lora_target_modules,
          lora_r=lora_r,
          wandb_project=wandb_project,
          micro_batch_size=micro_batch_size)


if __name__ == '__main__':
    prompt = {
        "description": "Template used by TAPIR-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further "
                        "context. Write a response that appropriately completes the request.\n\n### Instruction:"
                        "\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately "
                           "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"
    }

    adapter_name = 'action'

    run_experiment(
        model='meta-llama/Llama-3.1-8B',
        data_path=f'data/dataset/train_{adapter_name}.json',
        prompt=prompt,
        num_epochs=1,
        cutoff_len=512,
        group_by_length=True,
        output_dir=f'adapter/LLAMA31/{adapter_name}_adapter',
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_r=64,
        wandb_project=f'LLAMA3.1_ORA_TAPIR_{adapter_name}_adapter',
        micro_batch_size=8)
