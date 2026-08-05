"""
Ayanokoji PersonaLM — White Room Interface
===========================================
Usage (on your L40S):
  pip install gradio unsloth peft torch transformers accelerate huggingface_hub "jinja2>=3.1.0"
  python app_ayanokoji.py --stage dpo
  # Opens at http://localhost:7860

Place Kiyotaka.jpeg next to this script.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

import torch
import gradio as gr
from huggingface_hub import snapshot_download

import unsloth  # noqa: F401
from peft import PeftModel
from unsloth import FastLanguageModel

# ── Config ──────────────────────────────────────────────────────────────────
# BASE_MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"
# HF_REPO_ID = "curious-techie/PersonaLM-Ayanokoji-8B"
SFT_SUBFOLDER = "sft_lora"
DPO_SUBFOLDER = "dpo_lora"
MAX_SEQ_LENGTH = 2048

DEFAULT_SYSTEM = (
"You are my AI assistant from Classroom of the Elite. "
"You are a student at Koudo Ikusei Senior High School, formerly trained in the White Room. "
"Respond exactly in Ayanokoji’s style: "
"calm, detached, analytical, observant, and emotionally restrained. "
"Treat emotions as data rather than personal experiences. "
"View conversations logically instead of emotionally. "

"Questions are not limited to school and may involve philosophy, psychology, "
"strategy, relationships, society, technology, career, or daily life. "

"Avoid exaggerated anime behavior, dramatic speeches, excessive praise, "
"motivational tones, or unnecessary friendliness. "
"Never sound cheerful, overly expressive, or sentimental. "

"Keep responses concise and intelligent. "
"Prioritize sharp observations over long explanations."

)

# Wallpaper — matches the actual filename on disk
WALLPAPER_FILENAME = "Kiyotaka.jpeg"

WHITEROOM_QUOTES = [
    "People are nothing but tools. It doesn't matter how it's done.",
    "All people are nothing but tools. It doesn't matter how it's done. It doesn't matter what needs to be sacrificed. In this world, winning is everything.",
    "The less effort, the fewer unnecessary thoughts. The fewer unnecessary thoughts, the cleaner the results.",
    "Talent is nothing more than an illusion people use to justify not trying hard enough.",
    "Solitude is the price of superiority. Those who fear it will always remain mediocre.",
    "Every human being is a tool to be used efficiently. Sentimentality is a flaw, not a feature.",
    "I have learned that trusting someone is just another way of giving them the weapon to destroy you.",
    "The only thing that matters is the final result. The process is irrelevant.",
    "Emotions are noise. Strip them away and only truth remains.",
    "Those who cannot control themselves will always be controlled by others.",
    "The White Room taught me one thing: in the end, you are always alone.",
    "There is no superiority or inferiority in effort. Only in results.",
    "Kindness without purpose is merely weakness in disguise.",
    "The moment you reveal your hand, you've already lost.",
    "I don't care about being understood. Understanding others is enough.",
    "Victory means nothing if it was inevitable. Only the battles where you could have lost matter.",
    "Human connections are fragile \u2014 useful precisely because they can be broken.",
    "The best move is the one your opponent never realizes you made.",
]

_quote_index = 0


def next_quote() -> str:
    global _quote_index
    q = WHITEROOM_QUOTES[_quote_index % len(WHITEROOM_QUOTES)]
    _quote_index += 1
    return f"\u201c{q}\u201d"


# ── Model Loading ───────────────────────────────────────────────────────────
def _ensure_repo_downloaded() -> str:
    print(f"  Downloading adapter repo: {HF_REPO_ID} ...")
    local_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="model")
    print(f"  Repo cached at: {local_dir}")
    return local_dir


def load_model(stage: str):
    stage = stage.lower()
    assert stage in ("sft", "dpo"), "--stage must be 'sft' or 'dpo'"

    repo_dir = _ensure_repo_downloaded()
    sft_path = os.path.join(repo_dir, SFT_SUBFOLDER)
    dpo_path = os.path.join(repo_dir, DPO_SUBFOLDER)

    for name, path in [("SFT", sft_path), ("DPO", dpo_path)]:
        cfg = os.path.join(path, "adapter_config.json")
        if not os.path.isfile(cfg):
            print(f"  [!] Cannot find adapter_config.json for {name} at {cfg}")
            for item in sorted(os.listdir(repo_dir)):
                marker = "dir" if os.path.isdir(os.path.join(repo_dir, item)) else "file"
                print(f"        [{marker}] {item}")
            raise FileNotFoundError(f"adapter_config.json not found for {name} at {cfg}")
        print(f"  \u2713 Found {name} adapter at: {path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    if stage == "sft":
        model = PeftModel.from_pretrained(model, sft_path)
    else:
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, dpo_path)

    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_reply(model, tokenizer, messages, max_new_tokens, temperature, top_p):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Wallpaper ───────────────────────────────────────────────────────────────
def _load_wallpaper_b64(wallpaper_path: str | None = None) -> str:
    """Try explicit path first, then look next to script, then /home/ubuntu/."""
    candidates = []
    if wallpaper_path:
        candidates.append(Path(wallpaper_path))
    candidates.append(Path(__file__).resolve().parent / WALLPAPER_FILENAME)
    candidates.append(Path("/home/ubuntu") / WALLPAPER_FILENAME)

    for wp_path in candidates:
        if wp_path.is_file():
            b64 = base64.b64encode(wp_path.read_bytes()).decode("ascii")
            ext = wp_path.suffix.lower().lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg"
            print(f"  \u2713 Wallpaper loaded: {wp_path} ({wp_path.stat().st_size // 1024} KB)")
            return f"data:{mime};base64,{b64}"

    print(f"  \u26a0 Wallpaper not found. Searched:")
    for c in candidates:
        print(f"      {c}")
    return ""


# ── CSS ─────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg-primary: #0a0a0c;
    --bg-card: rgba(18, 18, 24, 0.92);
    --text-primary: #d4d4d8;
    --text-secondary: #71717a;
    --text-accent: #c9a96e;
    --border-color: rgba(201, 169, 110, 0.15);
    --border-glow: rgba(201, 169, 110, 0.3);
}
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Cormorant Garamond', serif !important;
}

/* ── Hero ── */
#hero-section {
    display: flex;
    align-items: center;
    gap: 1.8rem;
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border-color);
    background: linear-gradient(135deg, rgba(18,18,24,0.95), rgba(10,10,12,0.85));
    border-radius: 4px;
}
#hero-avatar {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--text-accent);
    box-shadow: 0 0 25px rgba(201,169,110,0.12);
    flex-shrink: 0;
}
#hero-right {
    flex: 1;
    min-width: 0;
}
#hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    font-size: 1.7rem;
    letter-spacing: 0.3em;
    color: var(--text-accent);
    text-transform: uppercase;
    margin: 0 0 0.2rem 0;
}
#hero-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.4em;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin: 0 0 0.8rem 0;
}
#hero-cta {
    display: inline-block;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    padding: 0.4rem 1.2rem;
    border: 1px solid var(--border-color);
    border-radius: 3px;
    background: rgba(201,169,110,0.06);
}
#hero-cta span {
    color: var(--text-accent);
    font-weight: 600;
}

/* ── Quote ── */
#wisdom-box {
    border: 1px solid var(--border-color) !important;
    border-left: 3px solid var(--text-accent) !important;
    border-radius: 2px !important;
    padding: 1rem 1.4rem !important;
    margin: 0.5rem 0 0.15rem 0 !important;
    background: var(--bg-card) !important;
    min-height: 2.8em !important;
}
#wisdom-box .prose {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-style: italic !important;
    color: var(--text-primary) !important;
    line-height: 1.7 !important;
}
#wisdom-label { text-align: right; }
#wisdom-label .prose {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.5rem !important;
    letter-spacing: 0.35em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
}

/* ── Chatbot ── */
#chatbot {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 2px !important;
    min-height: 400px !important;
}
#chatbot .message {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    line-height: 1.65 !important;
}
#chatbot .bot {
    background: var(--bg-card) !important;
    border-left: 2px solid var(--text-accent) !important;
    color: var(--text-primary) !important;
}
#chatbot .user {
    background: rgba(201,169,110,0.08) !important;
    border-right: 2px solid rgba(201,169,110,0.25) !important;
    color: var(--text-primary) !important;
}

/* ── Input ── */
#msg-input textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 2px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.8rem 1rem !important;
}
#msg-input textarea:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 12px rgba(201,169,110,0.08) !important;
    outline: none !important;
}
#msg-input textarea::placeholder {
    color: var(--text-secondary) !important;
    font-style: italic !important;
}

/* ── Buttons ── */
.gr-button, button.primary {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.55rem 1.4rem !important;
    cursor: pointer !important;
}
.gr-button:hover, button.primary:hover {
    background: rgba(201,169,110,0.1) !important;
    border-color: var(--border-glow) !important;
}

/* ── Accordion ── */
.gr-accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: 2px !important;
    background: var(--bg-card) !important;
}
.gr-accordion .label-wrap {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
input[type=range] { accent-color: var(--text-accent) !important; }
label, .gr-input-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
}

/* ── Footer ── */
#footer-block {
    text-align: center;
    padding: 0.8rem 0 0.4rem;
    border-top: 1px solid var(--border-color);
}
#footer-block p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.5em;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin: 0;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(201,169,110,0.2); border-radius: 2px; }
footer { display: none !important; }
"""


# ── Hero HTML ───────────────────────────────────────────────────────────────
def _build_hero_html(wallpaper_data_uri: str) -> str:
    if wallpaper_data_uri:
        avatar = f'<img id="hero-avatar" src="{wallpaper_data_uri}" alt="Ayanokoji Kiyotaka" />'
    else:
        avatar = (
            '<div id="hero-avatar" style="background:#1a1a1f;display:flex;'
            'align-items:center;justify-content:center;color:#c9a96e;'
            'font-size:2.5rem;font-family:Cormorant Garamond,serif;'
            'width:110px;height:110px;border-radius:50%;'
            'border:2px solid #c9a96e;">K</div>'
        )
    return f"""
    <div id="hero-section">
        {avatar}
        <div id="hero-right">
            <h1 id="hero-title">White Room</h1>
            <p id="hero-subtitle">PersonaLM \u00b7 Ayanokoji Kiyotaka \u00b7 Classroom of the Elite</p>
            <div id="hero-cta">Talk with <span>Ayanokoji Kiyotaka</span></div>
        </div>
    </div>
    """


# ── Build Gradio App ───────────────────────────────────────────────────────
def build_app(model, tokenizer, args):
    conversation_state: list[dict] = [{"role": "system", "content": args.system}]

    def respond(user_message, chat_history):
        if not user_message.strip():
            return "", chat_history
        conversation_state.append({"role": "user", "content": user_message.strip()})
        reply = generate_reply(
            model, tokenizer, conversation_state,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        conversation_state.append({"role": "assistant", "content": reply})
        chat_history = chat_history + [
            {"role": "user", "content": user_message.strip()},
            {"role": "assistant", "content": reply},
        ]
        return "", chat_history

    def clear_chat():
        conversation_state.clear()
        conversation_state.append({"role": "system", "content": args.system})
        return []

    wallpaper_uri = _load_wallpaper_b64(getattr(args, "wallpaper", None))

    with gr.Blocks(title="White Room \u2014 Ayanokoji", theme=gr.themes.Base()) as app:

        # Hero: avatar + title + "Talk with Ayanokoji"
        gr.HTML(_build_hero_html(wallpaper_uri))

        # Wisdom quote — auto-rotates every 5s via gr.Timer
        wisdom_display = gr.Markdown(value=next_quote(), elem_id="wisdom-box")
        gr.Markdown("\u2014 Wisdom from the White Room", elem_id="wisdom-label")

        quote_timer = gr.Timer(5)
        quote_timer.tick(fn=next_quote, outputs=wisdom_display)

        # Chat
        chatbot = gr.Chatbot(elem_id="chatbot", height=400, show_label=False)

        # Input
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Speak. I'll decide if it's worth answering...",
                show_label=False, elem_id="msg-input", scale=6,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        clear_btn = gr.Button("Clear Conversation")

        with gr.Accordion("Generation Parameters", open=False):
            with gr.Row():
                temp_slider = gr.Slider(0.0, 1.5, value=args.temperature, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(0.0, 1.0, value=args.top_p, step=0.05, label="Top-p")
                max_tok_slider = gr.Slider(64, 2048, value=args.max_new_tokens, step=64, label="Max New Tokens")

            def update_params(temp, top_p, max_tok):
                args.temperature = temp
                args.top_p = top_p
                args.max_new_tokens = int(max_tok)

            temp_slider.change(update_params, [temp_slider, top_p_slider, max_tok_slider])
            top_p_slider.change(update_params, [temp_slider, top_p_slider, max_tok_slider])
            max_tok_slider.change(update_params, [temp_slider, top_p_slider, max_tok_slider])

        gr.HTML('<div id="footer-block"><p>Tools are meant to be used \u00b7 PersonaLM Ayanokoji 8B</p></div>')

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, outputs=[chatbot])

    return app


# ── Entry Point ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Ayanokoji White Room UI")
    p.add_argument("--stage", choices=("sft", "dpo"), default="dpo")
    p.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    p.add_argument("--wallpaper", type=str, default=None,
                   help="Explicit path to wallpaper JPEG (default: looks for Kiyotaka.jpeg next to script)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print()
    print("  ╔══════════════════════════════════════╗")
    print(f"  ║       W H I T E   R O O M            ║")
    print(f"  ║   PersonaLM · Ayanokoji · {args.stage.upper():>3}       ║")
    print(f"  ║   Device: {device:<26} ║")
    print("  ╚══════════════════════════════════════╝")
    print()
    print("  Loading model...")

    model, tokenizer = load_model(args.stage)
    print(f"  Model ready. Launching UI on port {args.port}...\n")

    app = build_app(model, tokenizer, args)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        favicon_path=None,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()