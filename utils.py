# utils.py
import torch
import numpy as np

def evaluate_model(model, board):
    # Implement your custom logic to evaluate the model
    # Return a score based on the model's performance
    board_flat = board.flatten().astype(float)
    model_input = torch.tensor([board_flat], dtype=torch.float32)
    with torch.no_grad():
        model_output = model(model_input)
    return model_output.item()
