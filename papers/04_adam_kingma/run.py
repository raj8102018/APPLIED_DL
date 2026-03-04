#boiler plate code generated
import torch
from optimizer import CustomAdam

# Setup identical starting parameters
x_custom = torch.tensor([10.0], requires_grad=True)
x_official = torch.tensor([10.0], requires_grad=True)

# Initialize optimizers
opt_custom = CustomAdam([x_custom], lr=0.1)
opt_official = torch.optim.Adam([x_official], lr=0.1)

# Run exactly 50 steps of optimization
for step in range(50):
    # Custom pass
    opt_custom.zero_grad()
    loss_custom = x_custom ** 2
    loss_custom.backward()
    opt_custom.step()

    # Official pass
    opt_official.zero_grad()
    loss_official = x_official ** 2
    loss_official.backward()
    opt_official.step()

    # The absolute difference between your math and PyTorch's C++ backend
    diff = torch.abs(x_custom - x_official).item()
    
    # Assert they remain identical within float32 tolerance
    assert diff < 1e-5, f"FATAL: Divergence at step {step}! Custom: {x_custom.item():.5f}, Official: {x_official.item():.5f}"

print(f"SUCCESS: Custom Adam perfectly matches PyTorch official implementation.")
print(f"Final value: {x_custom.item():.5f} (Target: 0.00000)")