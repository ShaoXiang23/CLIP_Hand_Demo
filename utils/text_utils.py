import numpy as np
from PIL import Image
import clip
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32", device=device)

joints = [
    'wrist',
    'thumb MCP', 'thumb PIP', 'thumb DIP', 'thumb fingertip',
    'index MCP', 'index PIP', 'index DIP', 'index fingertip',
    'middle MCP', 'middle PIP', 'middle DIP', 'middle fingertip',
    'ring MCP', 'ring PIP', 'ring DIP', 'ring fingertip',
    'little MCP', 'little PIP', 'little DIP', 'little fingertip'
]

mano_joints = [
  'wrist',
  'index MCP', 'index PIP', 'index DIP',
  'middle MCP', 'middle PIP', 'middle DIP',
  'little MCP', 'little PIP', 'little DIP',
  'ring MCP', 'ring PIP', 'ring DIP',
  'thumb MCP', 'thumb PIP', 'thumb DIP',
  'index fingertip', 'middle fingertip', 'little fingertip', 'ring fingertip', 'thumb fingertip'
]