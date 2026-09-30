import importlib.util as u
for m in ["vllm_flash_attn", "flash_attn", "xformers", "vllm"]:
    print(m, "OK" if u.find_spec(m) else "MISSING")
import torch
print("torch", torch.__version__)
for path, name in [
    ("vllm.vllm_flash_attn", "flash_attn_varlen_func"),
    ("vllm_flash_attn", "flash_attn_varlen_func"),
    ("flash_attn", "flash_attn_varlen_func"),
]:
    try:
        mod = __import__(path, fromlist=[name])
        getattr(mod, name)
        print(f"USE: from {path} import {name}  -> OK")
    except Exception as e:
        print(f"{path}.{name}: {type(e).__name__}")
from torch.nn.attention import SDPBackend
print("SDPA backends available:", [b.name for b in SDPBackend])
