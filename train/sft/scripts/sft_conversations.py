import sys
import os
# Injects project root path so Python can resolve the local 'src' directory on Windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from transformers import AutoTokenizer

from src import get_checkpoint
from src.chat_templates import LLAMA_3_TEMPLATE
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    apply_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        print(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype="auto",
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA_3_TEMPLATE
        tokenizer.eos_token = "<|eot_id|>"

    tokenizer.pad_token = "<|endoftext|>"

    # This creates a DatasetDict containing a 'train' split key
    dataset = load_dataset("json", data_files="dummy_train.jsonl")

    def process_dataset(example, tokenizer=None):
        messages = example["messages"]
        return apply_chat_template({"messages": messages}, processing_class=tokenizer)

    processed_dataset = dataset.map(
        process_dataset, 
        num_proc=1,
        fn_kwargs={"tokenizer": tokenizer}
    )
    
    # Extract the column names strictly from the nested "train" Dataset split object
    columns_to_remove = processed_dataset["train"].column_names

    # Explicit helper function to prevent lambda multi-processing blindspots on Windows
    def tokenize_function(example, tokenizer=None):
        return tokenizer(example["text"], truncation=False)

    # Tokenize safely within a single process context
    processed_dataset = processed_dataset.map(
        tokenize_function,
        num_proc=1,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=columns_to_remove,
    )

    # Filter out examples longer than 512 tokens manually on the train split directly
    num_rows = processed_dataset["train"].num_rows
    processed_dataset["train"] = processed_dataset["train"].filter(
        lambda x: len(x["input_ids"]) <= 512, num_proc=1
    )
    print(
        f"Filtered {num_rows - processed_dataset['train'].num_rows} examples longer than 512 tokens."
    )

    print(f"=== FORMATTED SAMPLE === \n{tokenizer.decode(processed_dataset['train'][0]['input_ids'])}")

    # Instead of the broken old collator, tell SFTConfig to handle response masking natively
    training_args.completion_only_loss = True
    # Provide the exact template format your test model expects to track assistant splits
    training_args.response_template = "<|im_start|>assistant\n"

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=(
            processed_dataset["train"] if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_config),
        processing_class=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)