from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

app = FastAPI()

# directory to save uploaded files
UPLOAD_DIRECTORY = "./uploaded_files"

# checke if directory exists, if not create it
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# client database
clients_db = {}


class RegisterModel(BaseModel):
    client_id: str
    client_secret: str

@app.post("/register")
async def register_client(item: RegisterModel):
    if item.client_id in clients_db:
        return {"status": "Client already registered"}
    clients_db[item.client_id] = item.client_secret
    return {"status": "Client registered successfully"}

def validate_client(client_id: str, client_secret: str):
    if client_id not in clients_db or clients_db[client_id] != client_secret:
        raise HTTPException(status_code=401, detail="Invalid client credentials")

@app.post("/upload/")
async def upload_file(model: UploadFile = File(...), client_id: str = Form(...), client_secret: str = Form(...)):
    validate_client(client_id, client_secret)
    file_location = os.path.join(UPLOAD_DIRECTORY, model.filename)
    with open(file_location, "wb") as f:
        f.write(await model.read())
    return {"info": f"file '{model.filename}' saved at '{file_location}'"}

@app.post("/download/{file_name}")
async def download_file(file_name: str, client_id: str, client_secret: str):
    validate_client(client_id, client_secret)
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)
    else:
        raise HTTPException(status_code=404, detail="File not found")