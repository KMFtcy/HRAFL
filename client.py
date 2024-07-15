import requests
import os
import argparse

# Server URL
SERVER_URL = "http://127.0.0.1:8000"

# Local file path for model weights
local_weight_file = "model_weights.pt"

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
    else:
        print("Failed to register client:", response.json())

def upload_weights(client_id, client_secret):
    """
    Upload model weights to the server.
    """
    url = f"{SERVER_URL}/upload/"
    with open(local_weight_file, "rb") as f:
        files = {"model": (os.path.basename(local_weight_file), f, "application/octet-stream")}
        data = {"client_id": client_id, "client_secret": client_secret}
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("Weights uploaded successfully!")
        else:
            print("Failed to upload weights:", response.json())

def download_weights(client_id, client_secret):
    """
    Download model weights from the server.
    """
    url = f"{SERVER_URL}/download/{os.path.basename(local_weight_file)}"
    params = {"client_id": client_id, "client_secret": client_secret}
    response = requests.post(url, params=params, stream=True)
    if response.status_code == 200:
        with open(local_weight_file + ".download", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Weights downloaded successfully!")
    else:
        print("Failed to download weights:", response.json())

def train_model():
    """
    Model training and saving weights to a file.
    """
    print("Training model...")
    # Training code goes here (this is just a placeholder)
    # train()
    print("Training completed!")

def main_loop(client_id, client_secret):
    """
    Main loop for training, uploading, and downloading weights.
    """
    while True:
        train_model()
        upload_weights(client_id, client_secret)
        download_weights(client_id, client_secret)
        break

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Client for training and uploading model weights.")
    parser.add_argument("--client_id", required=True, help="Client ID for registration and authentication")
    parser.add_argument("--client_secret", required=True, help="Client Secret for registration and authentication")

    args = parser.parse_args()

    # Register the client
    register_client(args.client_id, args.client_secret)
    # Enter the training-upload-download loop
    main_loop(args.client_id, args.client_secret)