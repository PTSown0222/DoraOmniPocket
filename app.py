import torch
import gradio as gr
from huggingface_hub import hf_hub_download
import pickle

# 1. Tải checkpoint & Tokenizer từ Hugging Face Repo của bạn
REPO_ID = "username/repo-name" # Thay tên repo của bạn vào đây

# Tải file weight & tokenizer
model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model_MHA.pt")
tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="my_vi_bpe_tokenizer/tokenizer.pkl")

# Nạp Tokenizer
with open(tokenizer_path, "rb") as f:
    enc = pickle.load(f)

# 2. Khởi tạo mô hình (Import DoraModel & BaseModelConfig từ code của bạn)
# config = BaseModelConfig()
# model = DoraModel(config, MultiHeadAttention)
# model.load_state_dict(torch.load(model_path, map_location="cpu"))
# model.eval()

def generate_text(prompt, max_new_tokens, temperature):
    if not prompt.strip():
        return "Vui lòng nhập văn bản đầu vào!"
    
    # Encode prompt
    input_ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long)
    
    # Generative pass
    # with torch.no_grad():
    #     out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Decode output
    # result = enc.decode(out_ids[0].tolist())
    
    # (Demo output giả lập)
    result = f"{prompt} ... [Mô hình đã sinh ra văn bản tiếp theo]"
    return result

# 3. Tạo giao diện UI với Gradio
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, placeholder="Nhập câu mở đầu...", label="Input Prompt"),
        gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Vietnamese Mini GPT - Text Generation",
    description="Demo sinh văn bản tiếng Việt sử dụng kiến trúc Transformer tùy chỉnh."
)

if __name__ == "__main__":
    demo.launch()