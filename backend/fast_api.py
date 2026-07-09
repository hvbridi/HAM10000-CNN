from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import torch
import torchvision
from torch import nn

model = torchvision.models.mobilenet_v2()
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280,7))
model.load_state_dict('best_model.pth')

app = FastAPI()

@app.get('/health')
def health():
    return r'100% healthy'

@app.post('/send_image')
async def send_image(file: UploadFile):
    content = await file.read()
    img = Image.open(io.BytesIO(content)) 
