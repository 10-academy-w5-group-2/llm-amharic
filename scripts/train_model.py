import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")
import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from utils.utils import compute_metrics, evaluate_model, generate_predictions

def train_model():
    # Load dataset
    dataset = load_dataset('csv', data_files='data/amharic_news.csv')['train']
    
    # Split dataset into train and test sets
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Define category mappings
    categories = ["Others", "Local News", "Sports", "Entertainment", "Business", "International News", "Politics"]
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    id_to_category = {idx: cat for cat, idx in category_to_id.items()}
    
    model_name = "rasyosef/bert-small-amharic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    
    # Set format for datasets
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label={i: lbl for i, lbl in enumerate(categories)},
        label2id={lbl: i for i, lbl in enumerate(categories)},
        device_map="cuda"
    )
    
    # Print model information
    embedding_layer = model.base_model.embeddings
    print(f"Embedding layer: {embedding_layer}")
    print(f"Embedding details: {embedding_layer.word_embeddings.weight.shape}")
    print(f"Model configuration: {model.config}")
    
    # Evaluate model before fine-tuning
    before_finetuning_predictions = generate_predictions(model, test_dataset, device="cuda", id_to_category=id_to_category, num_samples=5)
    print(before_finetuning_predictions)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_before_metrics = evaluate_model(model, test_dataset, data_collator, device, 'test')
    print(test_before_metrics)
    
    train_before_metrics = evaluate_model(model, train_dataset, data_collator, device, "train")
    print(train_before_metrics)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_name + "-finetuned",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate model after fine-tuning
    test_after_metrics = evaluate_model(model, test_dataset, data_collator, device, 'test')
    print(test_after_metrics)
    
    train_after_metrics = evaluate_model(model, train_dataset, data_collator, device, "train")
    print(train_after_metrics)
    
    # Save the fine-tuned model
    trainer.save_model('./models/fine_tuned_model')
    tokenizer.save_pretrained('./models/fine_tuned_model')

if __name__ == "__main__":
    train_model()
