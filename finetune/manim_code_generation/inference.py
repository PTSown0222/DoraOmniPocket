import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "TheSon2202/mistral-manim-python-coder-v01"

MISTRAL_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ 'System: ' + message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + message['content'] + eos_token }}"
        "{% endif %}"
    "{% endfor %}"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Manim Code Generation Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model ID or path to local checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Create a square with side length 4 and color it red, then animate it to shift right by 3 units.",
        help="Instruction prompt for Manim animation"
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    return parser.parse_args()

def generate_manim_code(model, tokenizer, instruction: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    system_prompt = "You are an expert Python developer specialized in Manim animation code. Read the instruction and write clean, executable Manim code."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    args = parse_args()
    print(f"[INFO] Loading tokenizer and model from: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    print(f"[INFO] Generating code for prompt: \"{args.prompt}\"")
    code_result = generate_manim_code(
        model=model,
        tokenizer=tokenizer,
        instruction=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print("\n" + "=" * 20 + " GENERATED MANIM CODE " + "=" * 20)
    print(code_result)
    print("=" * 62)

if __name__ == "__main__":
    main()