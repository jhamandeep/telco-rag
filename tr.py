import torch
print("torch:", torch.__version__)                       # want ...+cu128
print("cuda available:", torch.cuda.is_available())      # True
print("device:", torch.cuda.get_device_name(0))          # NVIDIA GeForce RTX 5070
print("capability:", torch.cuda.get_device_capability(0))# (12, 0)  <- sm_120
x = torch.randn(4096, 4096, device="cuda")
y = x @ x                                                 # the REAL test
torch.cuda.synchronize()
print("matmul on GPU: OK", tuple(y.shape), y.device)