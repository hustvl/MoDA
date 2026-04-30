import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from bla_gpt import GPT, GPTConfig
from train_blagpt import DistributedDataLoader

@dataclass
class Hyperparameters:
    input_bin: str = '../data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = '../data/fineweb10B/fineweb_val_*.bin'
    batch_size: int = 8 * 64
    device_batch_size: int = 32
    sequence_length: int = 1024
    num_iterations: int = 5100
    learning_rate: float = 0.0036
    warmup_iters: int = 0
    warmdown_iters: int = 1450
    weight_decay: float = 0
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 1000

args = Hyperparameters()

assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)

val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, n_query_groups=3))
model = model.cuda()
model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

checkpoint_path = '/home/ubuntu/BlaGPT/bla_gpt/logs/2a022919-21e4-44bd-b052-ab571e7ac66b/state_step002000.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
raw_model.load_state_dict(checkpoint['model'])

val_loader.reset()
val_loss = 0.0
for _ in range(val_steps):
    x_val, y_val = val_loader.next_batch()
    with ctx:
        _, loss = model(x_val, y_val, return_logits=False)
        val_loss += loss.detach()
        del loss
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_loss /= val_steps

if master_process:
    print(f'Validation loss: {val_loss:.4f}')

dist.destroy_process_group()