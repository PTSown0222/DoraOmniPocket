import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
 
# Initialize the distributed environment
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
device = torch.device(f"cuda:{local_rank}")
print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}. Using device: {device}")
 
# Create pretraining model with default config, then wrap it in DDP
model_config = LlamaConfig()
model = LlamaForPretraining(model_config).to(rank)
model = DDP(model, device_ids=[local_rank])  # , output_device=local_rank)
model.train()