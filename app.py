import base64
import csv
import io
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter   # small runtime (used on Render)
except ImportError:
    import tensorflow as tf                              # fallback: TensorFlow's built-in copy
    Interpreter = tf.lite.Interpreter

APP_DIR = Path(__file__).parent          # the folder that contains app.py
IMG_DIR = APP_DIR / "test_images"
N_ROUNDS = 10                            # faces per game
DISPLAY_SIZE = 448                       # on-screen photo size in pixels

config = json.loads((APP_DIR / "model_config.json").read_text())
MODEL_NAME = config["model_name"]
MODEL_LABEL = {"MobileNet-FT": "fine-tuned MobileNet", "CNN+Aug": "CNN"}.get(MODEL_NAME, MODEL_NAME)
INPUT_SCALE = config["input_scale"]      # 255.0 for MobileNet, 1.0 for MLP/CNN
IMG_SIZE = config["img_size"]

# ---- The model: a LiteRT (.tflite) copy of best_model.keras, small enough for Render's free tier ----
interpreter = Interpreter(model_path=str(APP_DIR / "best_model.tflite"))
interpreter.allocate_tensors()
INPUT = interpreter.get_input_details()[0]
OUTPUT = interpreter.get_output_details()[0]


def predict_real_prob(batch):
    """P(Real) for each image, one image at a time (the model expects shape (1, 128, 128, 3))."""
    probs = []
    for img in batch:
        interpreter.set_tensor(INPUT["index"], img[None].astype(np.float32))
        interpreter.invoke()
        probs.append(float(interpreter.get_tensor(OUTPUT["index"]).ravel()[0]))
    return np.array(probs)

# ---- Load all images once and let the model predict them all up front ----
with open(IMG_DIR / "labels.csv") as f:
    rows = list(csv.DictReader(f))

LABELS = np.array([int(r["label"]) for r in rows])      # 1 = Real, 0 = AI
IMAGES = np.stack([
    np.array(Image.open(IMG_DIR / r["filename"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    for r in rows
])                                                       # uint8 pixels

x = IMAGES.astype("float32") / 255.0 * INPUT_SCALE     # same preprocessing as the notebook
MODEL_PROBS = predict_real_prob(x)                      # P(Real) for every image
MODEL_PREDS = (MODEL_PROBS > 0.5).astype(int)
print(f"Loaded {len(LABELS)} images | model accuracy on them: {np.mean(MODEL_PREDS == LABELS):.1%}")


@lru_cache(maxsize=None)
def photo_uri(idx):
    """Enlarge one 128px face for display and embed it as a data URI (the model still sees the original)."""
    img = Image.fromarray(IMAGES[idx]).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- Look and feel ----
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@400;500;600;700&family=Archivo+Narrow:wght@600;700&display=swap">')

CSS = """
:root { --real: #1f8a5b; --fake: #c8323c; }
.dark { --real: #3fb57f; --fake: #ef6b73; }

#intro *, #photo *, #board *, #verdict * { color: inherit; }
#intro, #board, #verdict, #photo { color: var(--body-text-color); }

#intro h1 { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700;
            font-size: 46px; line-height: 1.05; letter-spacing: -0.01em; margin: 0 0 8px; }
#intro p  { font-size: 17px; line-height: 1.5; max-width: 62ch; margin: 0;
            color: var(--body-text-color-subdued); }

/* The photo, presented as a print */
#photo .print { position: relative; width: min(100%, 472px); margin: 0 auto; padding: 12px;
                background: #ffffff; border-radius: 3px;
                box-shadow: 0 1px 2px rgba(0,0,0,.25), 0 10px 28px rgba(0,0,0,.18); }
#photo .print img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; }

/* The verdict stamp: the one bold moment */
#photo .stamp { position: absolute; top: 9%; right: 7%; transform: rotate(-9deg);
                font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700;
                font-size: clamp(44px, 8vw, 80px); line-height: 1; letter-spacing: 0.04em; padding: 6px 18px 4px;
                border: 6px double currentColor; border-radius: 8px;
                background: rgba(255,255,255,.88); animation: stamp-in .26s ease-out; }
#photo .stamp.real { color: #1f8a5b; }
#photo .stamp.fake { color: #c8323c; }
@keyframes stamp-in { from { transform: rotate(-9deg) scale(1.7); opacity: 0; }
                      to   { transform: rotate(-9deg) scale(1);   opacity: 1; } }
@media (prefers-reduced-motion: reduce) { #photo .stamp { animation: none; } }

/* Final result, same footprint as the photo so nothing jumps */
#photo .final { width: min(100%, 472px); aspect-ratio: 1 / 1; margin: 0 auto; box-sizing: border-box;
                display: flex; flex-direction: column; justify-content: center; padding: 40px;
                border: 1px solid var(--border-color-primary); border-radius: 3px;
                background: var(--block-background-fill); }
#photo .final h2 { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700;
                   font-size: 56px; line-height: 1; margin: 0 0 12px; }
#photo .final .score { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700;
                       font-size: 96px; line-height: 1; font-variant-numeric: tabular-nums; margin: 0 0 16px; }
#photo .final .score .to { font-size: 32px; font-weight: 600; margin: 0 12px;
                           color: var(--body-text-color-subdued); }
#photo .final p { font-size: 17px; line-height: 1.5; margin: 0; color: var(--body-text-color-subdued); }

/* Scoreboard: one dot per face */
#board .face-count { font-size: 15px; margin: 0 0 6px; color: var(--body-text-color-subdued); }
#board .track { display: grid; grid-template-columns: 64px 1fr auto; align-items: center; gap: 14px;
                padding: 10px 0; border-bottom: 1px solid var(--border-color-primary); }
#board .who { font-size: 16px; font-weight: 600; }
#board .num { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700; font-size: 44px;
              line-height: 1; min-width: 1.4ch; text-align: right; font-variant-numeric: tabular-nums; }
#board .dots { display: flex; flex-wrap: nowrap; gap: clamp(4px, 1.1vw, 8px); }
#board .dot { width: clamp(11px, 3vw, 17px); height: clamp(11px, 3vw, 17px); flex: none; border-radius: 50%; box-sizing: border-box;
              border: 2px solid var(--border-color-primary); }
#board .dot.now  { border-color: var(--body-text-color); }
#board .dot.hit  { background: var(--real); border-color: var(--real); }
#board .dot.miss { background: transparent; border: clamp(3px, 0.9vw, 4px) solid var(--fake); }

/* What happened this round */
#verdict { min-height: 118px; }
#verdict p { font-size: 17px; line-height: 1.55; margin: 0; }
#verdict .truth { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
#verdict .truth.real { color: var(--real); }
#verdict .truth.fake { color: var(--fake); }
#verdict .hint { color: var(--body-text-color-subdued); }

button.guess { font-size: 19px !important; font-weight: 600 !important; min-height: 68px !important; }
@media (max-width: 480px) { button.guess { font-size: 16px !important; min-height: 56px !important; } }
button.next  { font-size: 17px !important; font-weight: 600 !important; min-height: 52px !important; }
"""

INTRO_HTML = f"""
<h1>Spot the fake</h1>
<p>Half of these faces are real photos and half were made by an AI. You and my {MODEL_LABEL} model
guess each one. After {N_ROUNDS} faces, whoever got more right wins.</p>
"""


def label_text(y):
    return "a real photo" if y == 1 else "AI-generated"


def photo_html(state):
    if state["round"] >= N_ROUNDS:
        return final_html(state)
    idx = state["order"][state["round"]]
    stamp = ""
    if state["answered"]:                                    # stamp the answer onto the photo
        real = LABELS[idx] == 1
        stamp = f'<div class="stamp {"real" if real else "fake"}">{"REAL" if real else "AI"}</div>'
    return (f'<div class="print"><img src="{photo_uri(idx)}" '
            f'alt="Face {state["round"] + 1} of {N_ROUNDS}">{stamp}</div>')


def dots(results, current):
    """One dot per face: filled = right, ring = wrong, outlined = current, faint = not played yet."""
    out = []
    for i in range(N_ROUNDS):
        if i < len(results):
            cls = "hit" if results[i] else "miss"
        elif i == current:
            cls = "now"
        else:
            cls = ""
        out.append(f'<span class="dot {cls}"></span>')
    return "".join(out)


def board_html(state):
    finished = state["round"] >= N_ROUNDS
    current = -1 if finished else state["round"]
    you = [r["human_ok"] for r in state["history"]]
    bot = [r["model_ok"] for r in state["history"]]
    heading = "Final score" if finished else f"Face {state['round'] + 1} of {N_ROUNDS}"
    return f"""
    <p class="face-count">{heading}</p>
    <div class="track"><span class="who">You</span><span class="dots">{dots(you, current)}</span>
      <span class="num">{state['human']}</span></div>
    <div class="track"><span class="who">Model</span><span class="dots">{dots(bot, current)}</span>
      <span class="num">{state['model']}</span></div>
    """


def verdict_html(state):
    if state["round"] >= N_ROUNDS:
        return f'<p class="hint">Press Play again for {N_ROUNDS} new faces.</p>'
    if not state["answered"]:
        return '<p class="hint">Real photo or AI-generated? The model has already made its guess.</p>'
    last = state["history"][-1]
    truth_cls = "real" if last["truth"] == 1 else "fake"
    you = "You got it." if last["human_ok"] else "You missed this one."
    bot = (f"The model guessed {label_text(last['model_guess'])}, {last['model_conf']:.0%} sure, "
           f"and was {'right' if last['model_ok'] else 'wrong'}.")
    return (f'<p class="truth {truth_cls}">It’s {label_text(last["truth"])}.</p>'
            f'<p>{you}</p><p>{bot}</p>')


def final_html(state):
    h, m = state["human"], state["model"]
    title = "You win" if h > m else ("The model wins" if m > h else "It’s a tie")
    return f"""
    <div class="final">
      <h2>{title}</h2>
      <div class="score">{h}<span class="to">to</span>{m}</div>
      <p>You spotted {h} of {N_ROUNDS} faces. The model spotted {m}.</p>
    </div>"""


# ---- Game logic ----
def render(state):
    """Turn the current game state into values for every on-screen component."""
    finished = state["round"] >= N_ROUNDS
    guessing = not state["answered"] and not finished
    last_round = state["round"] == N_ROUNDS - 1
    return (
        state,
        photo_html(state),
        board_html(state),
        verdict_html(state),
        gr.Button(interactive=guessing),                                  # Real photo
        gr.Button(interactive=guessing),                                  # AI-generated
        gr.Button(value="Play again" if finished else ("See who won" if last_round else "Next face"),
                  interactive=state["answered"] or finished),             # Next / Play again
    )


def new_game():
    state = {
        "order": random.sample(range(len(LABELS)), N_ROUNDS),   # different random faces each game
        "round": 0,
        "human": 0,
        "model": 0,
        "answered": False,
        "history": [],
    }
    return render(state)


def make_guess(choice, state):
    if state["answered"] or state["round"] >= N_ROUNDS:          # ignore extra clicks
        return render(state)
    idx = state["order"][state["round"]]
    truth = int(LABELS[idx])
    model_guess = int(MODEL_PREDS[idx])
    p = float(MODEL_PROBS[idx])
    state["history"].append({
        "truth": truth,
        "human_ok": choice == truth,
        "model_ok": model_guess == truth,
        "model_guess": model_guess,
        "model_conf": p if model_guess == 1 else 1 - p,           # how sure the model was
    })
    state["human"] += int(choice == truth)
    state["model"] += int(model_guess == truth)
    state["answered"] = True
    return render(state)


def next_round(state):
    if state["round"] >= N_ROUNDS:                               # after the last face: new game
        return new_game()
    if state["answered"]:
        state["round"] += 1
        state["answered"] = False
    return render(state)


# ---- Layout ----
with gr.Blocks(title="Spot the fake") as demo:
    gr.HTML(INTRO_HTML, elem_id="intro")
    state = gr.State()

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=320):
            photo = gr.HTML(elem_id="photo")
        with gr.Column(scale=5, min_width=320):
            board = gr.HTML(elem_id="board")
            with gr.Row():
                real_btn = gr.Button("Real photo", elem_classes="guess", interactive=False)
                ai_btn = gr.Button("AI-generated", elem_classes="guess", interactive=False)
            verdict = gr.HTML(elem_id="verdict")
            next_btn = gr.Button("Next face", variant="primary", elem_classes="next", interactive=False)
            restart_btn = gr.Button("Start a new game", variant="secondary", size="sm")

    # Every function returns values for all of these, in this order
    outputs = [state, photo, board, verdict, real_btn, ai_btn, next_btn]

    demo.load(new_game, inputs=None, outputs=outputs)                          # start when the page opens
    real_btn.click(lambda s: make_guess(1, s), inputs=state, outputs=outputs)   # 1 = Real
    ai_btn.click(lambda s: make_guess(0, s), inputs=state, outputs=outputs)     # 0 = AI
    next_btn.click(next_round, inputs=state, outputs=outputs)
    restart_btn.click(new_game, inputs=None, outputs=outputs)

THEME = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Archivo"), "system-ui", "sans-serif"],
)

if __name__ == "__main__":
    on_render = "RENDER" in os.environ                   # Render sets this variable automatically
    demo.launch(
        theme=THEME, css=CSS, head=FONTS,                # Gradio 6: theme, css and head go in launch()
        server_name="0.0.0.0" if on_render else "127.0.0.1",   # Render needs outside connections
        server_port=int(os.environ.get("PORT", 7860)),         # Render tells the app which port to use
    )