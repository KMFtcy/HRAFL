import requests
import os
import argparse
import importlib
import time
import shutil
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from peft import PeftModel
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from transformers import get_linear_schedule_with_warmup
import os
from transformers import AdamW
os.environ["WANDB_MODE"] = "dryrun"

# Server URL
SERVER_URL = "http://127.0.0.1:8000"

# Local file path for model weights
local_weight_dir = ""

# Training settings
train_setting = {}

def check_download_status(client_id, client_secret, token):
    """
    Check if the model is ready to be downloaded from the server.
    """
    url = f"{SERVER_URL}/check_download"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "token": token
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["status"]
    else:
        print("Failed to check download status:", response.json())
        return 1001

def register_client(client_id, client_secret):
    """
    Register the client with the server.
    """
    url = f"{SERVER_URL}/register"
    client_info = {
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, json=client_info)
    if response.status_code == 200:
        print("Client registered successfully!")
        train_setting["token"] = response.json()["data"]["token"]
    else:
        print("Failed to register client:", response.json())

def upload_weights(client_id, client_secret):
    """
    Uploads the entire weights directory to the server as a zip file.
    """
    # Create a zip file of the local_weight_dir
    zip_filename = os.path.basename(local_weight_dir) + ".zip"
    shutil.make_archive(os.path.splitext(zip_filename)[0], 'zip', local_weight_dir)

    # Upload the zip file
    url = f"{SERVER_URL}/upload/"
    with open(zip_filename, "rb") as f:
        files = {"model": (zip_filename, f, "application/zip")}
        data = {"client_id": client_id, "client_secret": client_secret}
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            print("Weights uploaded successfully!")
            train_setting["token"] = response.json()["data"]["token"]
        else:
            print("Failed to upload weights:", response.json())

    # Clean up the zip file after upload
    os.remove(zip_filename)

def download_weights(client_id, client_secret, token):
    """
    Download model weights from the server.
    """
    url = f"{SERVER_URL}/download"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "token": token
    }
    response = requests.get(url, params=params, stream=True)
    if response.status_code == 200:
        if response.headers.get('content-type') == 'application/json':
            print("No merged model available for token", token)
            print(response.json())
        else:
            with open(os.path.join(local_weight_dir , f"adapter_model.safetensors"), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Weights downloaded successfully!")
    else:
        print("Failed to download weights:", response.json())

def import_train_function(file_path):
    # Extract the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the train function from the module
    train_function = getattr(module, 'train', None)

    if train_function is None:
        raise AttributeError(f"The module '{module_name}' does not have a 'train' function.")

    return train_function

def train():
    model = AutoModelForSequenceClassification.from_pretrained(train_setting["model_checkpoint"], num_labels=2)
    train_setting["peft_config"] = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    if not os.path.exists(local_weight_dir):
        model = get_peft_model(model, train_setting["peft_config"])
    else:
        print("Found local weights, loading from", local_weight_dir)
        model = PeftModel.from_pretrained(model, local_weight_dir)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    training_args = TrainingArguments(
        output_dir=os.path.join("output_results", str(train_setting["token"]), local_weight_dir),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join("logs", str(train_setting["token"]), local_weight_dir),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        max_grad_norm=1.0
    )

    # create learning rate adaptor
    num_training_steps = len(train_setting["dataset"]['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_setting["tokenized_train_dataset"],
        eval_dataset=train_setting["tokenized_eval_dataset"],
    )

    # trainer.train()

    trainer.save_model(local_weight_dir)

    return trainer

def main_loop(client_id, client_secret):
    """
    Main loop for training, uploading, and downloading weights.
    """
    train_setting["dataset"] = load_dataset("imdb")
    train_setting["train_dataset"] = train_setting["dataset"]["train"]
    train_setting["eval_dataset"] = train_setting["dataset"]["test"]
    train_setting["model_checkpoint"] = "bert-base-uncased"
    train_setting["tokenizer"] = AutoTokenizer.from_pretrained(train_setting["model_checkpoint"])
    def preprocess_function(examples):
        return train_setting["tokenizer"](examples["text"], truncation=True, padding="max_length")
    train_setting["tokenized_train_dataset"] = train_setting["train_dataset"].map(preprocess_function, batched=True)
    train_setting["tokenized_eval_dataset"] = train_setting["eval_dataset"].map(preprocess_function, batched=True)

    while True:
        train()
        # trainer = train_func(training_args, "bert-base-uncased", local_weight_dir)
        upload_weights(client_id, client_secret)

        # Polling to check if the model is ready for download
        while True:
            status = check_download_status(client_id, client_secret, train_setting["token"])
            if status == 1000:
                print("Weights aggregated successfully!")
                download_weights(client_id, client_secret, train_setting["token"])
                break
            else:
                print("Model not ready, waiting 1 second...")
                time.sleep(1)

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Client for training and uploading model weights.")
    parser.add_argument("--client_id", required=True, help="Client ID for registration and authentication")
    parser.add_argument("--client_secret", required=True, help="Client Secret for registration and authentication")
    parser.add_argument("--train_script", required=True, help="Client Train Script")

    args = parser.parse_args()

    # Register the client
    register_client(args.client_id, args.client_secret)
    # set global setting
    local_weight_dir = args.client_id + "_train"
    # Enter the training-upload-download loop
    main_loop(args.client_id, args.client_secret)