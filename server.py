from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import torch
import os

app = FastAPI()

# check if directory exists, if not create it
UPLOAD_DIRECTORY = "./uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# client database
clients_db = {}
upload_record = {}


def average_lora_models(model_paths, weights):
    # 加载第一个模型的state_dict
    state_dict_a = torch.load(model_paths[0], map_location='cpu')
    # 初始化一个字典来存储加权后的参数
    averaged_state_dict = {k: weights[0] * v for k, v in state_dict_a.items()}
    # 对于每个额外的模型
    for i in range(1, len(model_paths)):
        state_dict_i = torch.load(model_paths[i], map_location='cpu')
        # 将当前模型的参数加权累加到averaged_state_dict中
        for k, v in state_dict_i.items():
            averaged_state_dict[k] += weights[i] * v
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
    model_paths = [os.path.join(UPLOAD_DIRECTORY, f'{client_id}_{current_token-1}.pt') for client_id in clients_db.keys()]
    weights = [1.0 / len(model_paths)] * len(model_paths)  # equal weights
    # averaged_state_dict = average_lora_models(model_paths, weights)
    averaged_state_dict = {}
    torch.save(averaged_state_dict, os.path.join(UPLOAD_DIRECTORY, f'merged_model_{current_token}.pt'))

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    model: UploadFile = File(...),
    client_id: str = Form(...),
    client_secret: str = Form(...)
):
    validate_client(client_id, client_secret)
    file_location = os.path.join(UPLOAD_DIRECTORY, f'{client_id}_{clients_db[client_id]["token"]}.pt')
    with open(file_location, "wb") as f:
        f.write(await model.read())
    old_token = clients_db[client_id]["token"]
    clients_db[client_id]["token"] = old_token + 1

    # try to merge models
    current_token = clients_db[client_id]["token"]
    if check_all_clients_uploaded(current_token):
        background_tasks.add_task(merge_models_task,current_token)

    return {
        "info": f"file '{model.filename}' saved at '{file_location}'",
        "data": clients_db[client_id]
    }

@app.get("/download")
async def download_file(token: int, client_id: str, client_secret: str):
    validate_client(client_id, client_secret)
    merged_file_path = os.path.join(UPLOAD_DIRECTORY, f'merged_model_{token}.pt')

    if os.path.exists(merged_file_path):
        return FileResponse(merged_file_path, media_type='application/octet-stream', filename=f'merged_model_{token}.pt')
    else:
        return JSONResponse(status_code=200, content={"status": 1001, "detail": f"No merged model available for token {token}"})

@app.get("/check_download")
async def check_download(token: int, client_id: str, client_secret: str):
    validate_client(client_id, client_secret)
    merged_file_path = os.path.join(UPLOAD_DIRECTORY, f'merged_model_{token}.pt')

    if os.path.exists(merged_file_path):
        return JSONResponse(status_code=200, content={"status": 1000})
    else:
        return JSONResponse(status_code=200, content={"status": 1001})