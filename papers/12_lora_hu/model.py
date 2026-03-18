import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):

    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:

        super().__init__()

        self.original_layer = original_layer

        for layer in original_layer.parameters():
            layer.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias= False)

        self.dropout = nn.Dropout(dropout)

        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        original_out = self.original_layer(x)

        x_dropout = self.dropout(x)
        lora_A_out = self.lora_A(x_dropout)
        lora_B_out = self.lora_B(lora_A_out)
        lora_output = self.scaling * lora_B_out

        return original_out + lora_output
    

   