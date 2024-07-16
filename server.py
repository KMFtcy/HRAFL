from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import torch
import os
import zipfile
import tempfile
import shutil
import safetensors.torch as storch

app = FastAPI()

# check if directory exists, if not create it
UPLOAD_DIRECTORY = "./uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
MERGED_MODEL_DIRECTORY = "./merged_models"
if not os.path.exists(MERGED_MODEL_DIRECTORY):
    os.makedirs(MERGED_MODEL_DIRECTORY)

# client database
clients_db = {}
upload_record = {}

def average_lora_models(model_dirs, weights):
    state_dict_a = storch.load_file(f"{model_dirs[0]}/adapter_model.safetensors")
    averaged_state_dict = {k: weights[0] * v for k, v in state_dict_a.items()}
    for i in range(1, len(model_dirs)):
        state_dict_i = storch.load_file(f"{model_dirs[i]}/adapter_model.safetensors")
        for k, v in state_dict_i.items():
            averaged_state_dict[k] = averaged_state_dict.get(k, 0) + weights[i] * v
    return averaged_state_dict

class RegisterRquestModel(BaseModel):
    client_id: str
    client_secret: str

@app.post("/register")
async def register_client(item: RegisterRquestModel):
    if item.client_id in clients_db:
        return {
            "status": "Client already registered",
            "data": clients_db[item.client_id]
        }
    clients_db[item.client_id] = {
        "client_secret": item.client_secret,
        "token": 0
    }
    return {
        "status": "Client registered successfully",
        "data": clients_db[item.client_id]
    }

def validate_client(client_id: str, client_secret: str):
    if client_id not in clients_db or clients_db[client_id]["client_secret"] != client_secret:
        raise HTTPException(status_code=401, detail="Invalid client credentials")

def check_all_clients_uploaded(current_token):
    for client in clients_db.values():
        if client["token"] < current_token:
            return False
    return True

def merge_models_task(current_token):
    # Get model paths for all clients for the previous token
    model_paths = [os.path.join(UPLOAD_DIRECTORY, client_id, str(current_token - 1)) for client_id in clients_db.keys()]
    weights = [1.0 / len(model_paths)] * len(model_paths)  # equal weights
    averaged_state_dict = average_lora_models(model_paths, weights)
    storch.save_file(averaged_state_dict, os.path.join(MERGED_MODEL_DIRECTORY, f'merged_model_{current_token}.safetensors'))

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    model: UploadFile = File(...),
    client_id: str = Form(...),
    client_secret: str = Form(...)
):
    validate_client(client_id, client_secret)

    # Create client-specific upload directory
    client_upload_directory = os.path.join(UPLOAD_DIRECTORY,client_id, str(clients_db[client_id]["token"]))
    os.makedirs(client_upload_directory, exist_ok=True)

    # Create a temporary directory to save the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, model.filename)
        with open(zip_path, "wb") as f:
            f.write(await model.read())

        # Extract files to the client-specific upload directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(client_upload_directory)

    # Update client's token
    old_token = clients_db[client_id]["token"]
    clients_db[client_id]["token"] = old_token + 1

    # Try to merge models
    current_token = clients_db[client_id]["token"]
    if check_all_clients_uploaded(current_token):
        background_tasks.add_task(merge_models_task, current_token)

    return {
        "info": f"file '{model.filename}' saved and extracted at '{client_upload_directory}'",
        "data": clients_db[client_id]
    }

@app.get("/download")
async def download_file(token: int, client_id: str, client_secret: str):
    validate_client(client_id, client_secret)
    merged_file_path = os.path.join(MERGED_MODEL_DIRECTORY, f'merged_model_{token}.safetensors')

    if os.path.exists(merged_file_path):
        return FileResponse(merged_file_path, media_type='application/octet-stream', filename=f'merged_model_{token}.safetensors')
    else:
        return JSONResponse(status_code=200, content={"status": 1001, "detail": f"No merged model available for token {token}"})

@app.get("/check_download")
async def check_download(token: int, client_id: str, client_secret: str):
    validate_client(client_id, client_secret)
    merged_file_path = os.path.join(MERGED_MODEL_DIRECTORY, f'merged_model_{token}.safetensors')

    if os.path.exists(merged_file_path):
        return JSONResponse(status_code=200, content={"status": 1000})
    else:
        return JSONResponse(status_code=200, content={"status": 1001})