import torch
from model import *

mod = OusterNetNarrow().eval()
example = torch.randn(1, 3, 128, 512)
traced_module = torch.jit.script(mod)
traced_module.save("lol_script.pt")
