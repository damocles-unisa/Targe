import time
import warnings

import pandas as pd
import evaluate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
warnings.simplefilter(action='ignore', category=UserWarning)

def load_model(adapter_model_id='adapter/recipe_adapter'):
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', )
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B',
        load_in_8bit=True,
        device_map="auto",
    )

    tapir_model = PeftModel.from_pretrained(model, model_id=adapter_model_id, is_trainable=True)

    print(tapir_model.print_trainable_parameters())

    return tapir_model, tokenizer

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}
        ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        ### Instruction:
        {instruction}
        ### Response:"""

if __name__ == '__main__':
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=0.9,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128001,
    )

    bleu = evaluate.load("bleu")

    em_value = 0
    bleu_value = 0

    set = 'action'
    spare = 'gold'
    df = pd.read_csv(f'data/dataset/{spare}/test_{set}.csv')
    model, tokenizer = load_model(f'adapter/LLAMA3/{set}_adapter')

    with tqdm(len(df.index), desc=f'Evaluating {set} {spare} adapter', total=len(df)) as pbar:
        start_time = time.time()
        pbar.set_postfix(BLEU=0.0, EM=0.0, Time_Left=0)

        predictions = []
        references = []

        for index, row in df.iterrows():
            instruction = row['instruction']
            input = row['input']
            expected_output = row['output']

            prompt = generate_prompt(instruction=instruction, input=input)
            inputs = tokenizer(prompt, return_tensors="pt")

            # Generate
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs.get("attention_mask", None)

            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_scores=True,
                max_new_tokens=256
            )

            output = tokenizer.batch_decode(generation_output, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0].split("### Response:")[1].strip()

            try:
                df.loc[index, 'generated'] = output

                # Update EM counter
                if output == expected_output:
                    em_value += 1

                # Add to BLEU computation lists
                predictions.append(output)  # BLEU expects tokenized sequences
                references.append([expected_output])  # BLEU expects a list of reference sequences

                elapsed_time = time.time() - start_time
                avg_time_per_sample = elapsed_time / (index + 1)
                remaining_time = avg_time_per_sample * (len(df.index) - (index + 1))
                bleu_score = bleu.compute(predictions=predictions, references=references)

                pbar.set_postfix(BLEU=bleu_score['bleu'], EM=em_value/(index + 1), Time_Left=f'{remaining_time:.2f}s')
                pbar.update(1)
            except Exception as e:
                df.loc[index, 'generated'] = output

            if index % 10 == 0:
                df.to_csv(f'data/generated/generated_{set}_{spare}.csv')


    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU: {bleu_score['bleu']}")
    print(f"EM: {em_value/len(df.index)}")
    df.to_csv(f'data/generated/generated_{set}_{spare}.csv')

    if set == 'trigger' or 'action':
        print("To evaluate the ICG version use the evaluate_llm.ipynb notebook")