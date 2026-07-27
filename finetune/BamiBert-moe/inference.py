
import os
from transformers import AutoConfig
from safetensors.torch import save_file

def average_checkpoints_to_safetensors(config, n_splits=5):
    export_dir = "/kaggle/working/bamibert_moe_final"
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"Processing Weighted Summation from {n_splits} Folds...")
    
    # Init Fold 1
    avg_state_dict = torch.load("bamibert_moe_fold1.pth", map_location='cpu')
    
    # Summation Fold 2 -> 5
    for i in range(2, n_splits + 1):
        fold_state_dict = torch.load(f"bamibert_moe_fold{i}.pth", map_location='cpu')
        for key in avg_state_dict:
            avg_state_dict[key] += fold_state_dict[key]
            
    # mean
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / n_splits
        
    safetensors_path = os.path.join(export_dir, "model.safetensors")
    save_file(avg_state_dict, safetensors_path)
    
    print(f"Summation Weights at: {safetensors_path}")
    return export_dir


final_folder = average_checkpoints_to_safetensors(config)
tokenizer.save_pretrained(final_folder)
config_hf = AutoConfig.from_pretrained(config.model_name)
config_hf.save_pretrained(final_folder)

print(f"\nDone: {final_folder}")

