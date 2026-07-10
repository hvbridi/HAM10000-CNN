from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import torch
import torchvision
from torch import nn
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware


weights = torch.load('best_model.pth')
model = torchvision.models.mobilenet_v2()
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280,7))
model.load_state_dict(weights)
model.eval()

transform = transforms.Compose([transforms.Resize((256)), transforms.CenterCrop(224),transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This allows your HTML file to connect
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get('/health')
def health():
    return r'100% healthy'

@app.post('/send_image')
async def send_image(file: UploadFile):
    content = await file.read()
    img = Image.open(io.BytesIO(content)) 
    img = img.convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.inference_mode():
        logits = model(img)
    probs = torch.softmax(logits,dim=1)
    max_prob, pred = torch.max(probs,dim=1)
    if max_prob > 0.8:
        return pred.item()
    return 99