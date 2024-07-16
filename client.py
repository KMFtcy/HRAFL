import requests
import os
import argparse
import importlib
import time
import shutil
from transformers import TrainingArguments, Trainer

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

def main_loop(client_id, client_secret, train_func):
    """
    Main loop for training, uploading, and downloading weights.
    """
    training_args = TrainingArguments(
        output_dir=os.path.join("output_results", client_id),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join("logs", client_id),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        max_grad_norm=1.0
    )
    while True:
        trainer = train_func(training_args)
        trainer.save_model(local_weight_dir)
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
    # Load train script
    train_func = import_train_function(args.train_script)
    # set local file name
    local_weight_dir = args.client_id + "_train"
    # Enter the training-upload-download loop
    main_loop(args.client_id, args.client_secret, train_func)