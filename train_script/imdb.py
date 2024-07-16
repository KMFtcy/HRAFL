from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from transformers import get_linear_schedule_with_warmup
import os
from transformers import AdamW
os.environ["WANDB_MODE"] = "dryrun"

def train(training_args):
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    model = get_peft_model(model, peft_config)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # create learning rate adaptor
    num_training_steps = len(dataset['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    trainer.train()

    # trainer.save_model("./my_lora_bert")

    return trainer

# verification
# classifier = pipeline("sentiment-analysis", model="./my_lora_bert")
# result = classifier("This is a great movie!")
# print(result)