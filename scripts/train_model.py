import os
from datasets import Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftModel

def train_model():
    # Load the tokenized dataset
    dataset_path = 'data/tokenized_dataset'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    dataset = Dataset.load_from_disk(dataset_path)
    print('The dataset is:', dataset)

    if len(dataset) == 0:
        raise ValueError("The loaded dataset is empty.")
    
    # Check and print dataset sample
    try:
        print('Sample data from dataset:', dataset[0])
    except IndexError as e:
        raise ValueError(f"Error accessing dataset sample: {e}")
    
    # Load the tokenizer and model
    MAIN_PATH = '/home/abraham_teka/Llama-2-7b-hf'
    peft_model_path = '/home/abraham_teka/llama-2-amharic-3784m/pretrained'
    model_name = MAIN_PATH

    model = load_model(model_name, True)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    embedding_size = model.get_input_embeddings().weight.shape[0]

    # Check tokenizer and embedding size
    if len(tokenizer) != embedding_size:
        print("Resizing the embedding size to match the tokenizer size")
        model.resize_token_embeddings(len(tokenizer))

    # Load the PEFT model if provided
    if peft_model_path:
        model = load_peft_model(model, peft_model_path, len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./models/fine_tuned_model',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=200,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # Define data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model('./models/fine_tuned_model')
    tokenizer.save_pretrained('./models/fine_tuned_model')

def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

def load_peft_model(model, peft_model_path, tokenizer_length):
    # Adjust the PEFT model to match the tokenizer length
    peft_model = PeftModel.from_pretrained(model, peft_model_path)
    peft_model.resize_token_embeddings(tokenizer_length)
    return peft_model

if __name__ == "__main__":
    train_model()
