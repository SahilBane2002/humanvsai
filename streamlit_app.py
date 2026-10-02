
import base64
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter   # small runtime, same as on Render
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

APP_DIR = Path(__file__).parent
IMG_DIR = APP_DIR / "test_images"
N_ROUNDS = 10            # faces per game
DISPLAY_SIZE = 448       # on-screen photo size in pixels

st.set_page_config(page_title="Spot the fake", layout="wide")


# ---- Model and images: loaded once per server, shared by every visitor ----
@st.cache_resource
def load_game_data():
    config = json.loads((APP_DIR / "model_config.json").read_text())
    with open(IMG_DIR / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["label"]) for r in rows])                      # 1 = Real, 0 = AI
    size = (config["img_size"], config["img_size"])
    images = np.stack([np.array(Image.open(IMG_DIR / r["filename"]).convert("RGB").resize(size))
                       for r in rows])                                       # uint8 pixels

    interpreter = Interpreter(model_path=str(APP_DIR / "best_model.tflite"))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    x = images.astype("float32") / 255.0 * config["input_scale"]            # same preprocessing as the notebook
    probs = []
    for img in x:                                                            # the model takes one image at a time
        interpreter.set_tensor(inp["index"], img[None])
        interpreter.invoke()
        probs.append(float(interpreter.get_tensor(out["index"]).ravel()[0]))
    probs = np.array(probs)                                                  # P(Real) for every image
    return config, labels, images, probs, (probs > 0.5).astype(int)


CONFIG, LABELS, IMAGES, MODEL_PROBS, MODEL_PREDS = load_game_data()
MODEL_LABEL = {"MobileNet-FT": "fine-tuned MobileNet", "CNN+Aug": "CNN"}.get(CONFIG["model_name"], CONFIG["model_name"])


@st.cache_data
def photo_uri(idx):
    """Enlarge one 128px face for display (the model still sees the original)."""
    img = Image.fromarray(IMAGES[idx]).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- Look and feel ----
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Archivo+Narrow:wght@600;700&display=swap');
:root { --real: #1f8a5b; --fake: #c8323c; --ink: #111827; --muted: #5b6472; --line: #d5dae1; }
.block-container { max-width: 1100px; padding-top: 2.2rem; }
.game, .game * { font-family: 'Archivo', system-ui, sans-serif; color: var(--ink); }
.game h1 { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700; font-size: 46px;
           line-height: 1.05; letter-spacing: -0.01em; margin: 0 0 8px; padding: 0; }
.game .intro { font-size: 17px; line-height: 1.5; max-width: 62ch; color: var(--muted); margin: 0 0 18px; }
.game .print { position: relative; width: min(100%, 472px); margin: 0 auto; padding: 12px; background: #fff;
               border-radius: 3px; box-shadow: 0 1px 2px rgba(0,0,0,.2), 0 10px 28px rgba(0,0,0,.14); }
.game .print img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; }
.game .stamp { position: absolute; top: 9%; right: 7%; transform: rotate(-9deg);
               font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700;
               font-size: clamp(44px, 8vw, 80px); line-height: 1; letter-spacing: 0.04em;
               padding: 6px 18px 4px; border: 6px double currentColor; border-radius: 8px;
               background: rgba(255,255,255,.88); animation: stamp-in .26s ease-out; }
.game .stamp.real { color: var(--real); }
.game .stamp.fake { color: var(--fake); }
@keyframes stamp-in { from { transform: rotate(-9deg) scale(1.7); opacity: 0; }
                      to   { transform: rotate(-9deg) scale(1);   opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .game .stamp { animation: none; } }
.game .final { width: min(100%, 472px); aspect-ratio: 1 / 1; margin: 0 auto; box-sizing: border-box;
               display: flex; flex-direction: column; justify-content: center; padding: 40px;
               border: 1px solid var(--line); border-radius: 3px; background: #fff; }
.game .final h2 { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700; font-size: 56px;
                  line-height: 1; margin: 0 0 12px; padding: 0; }
.game .final .score { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700; font-size: 96px;
                      line-height: 1; margin: 0 0 16px; }
.game .final .score .to { font-size: 32px; font-weight: 600; margin: 0 12px; color: var(--muted); }
.game .final p { font-size: 17px; line-height: 1.5; margin: 0; color: var(--muted); }
.game .face-count { font-size: 15px; margin: 0 0 6px; color: var(--muted); }
.game .track { display: grid; grid-template-columns: 64px 1fr auto; align-items: center; gap: 14px;
               padding: 10px 0; border-bottom: 1px solid var(--line); }
.game .who { font-size: 16px; font-weight: 600; }
.game .num { font-family: 'Archivo Narrow', 'Archivo', sans-serif; font-weight: 700; font-size: 44px;
             line-height: 1; min-width: 1.4ch; text-align: right; }
.game .dots { display: flex; flex-wrap: nowrap; gap: clamp(4px, 1.1vw, 8px); }
.game .dot { width: clamp(11px, 3vw, 17px); height: clamp(11px, 3vw, 17px); flex: none; border-radius: 50%;
             box-sizing: border-box; border: 2px solid var(--line); }
.game .dot.now  { border-color: var(--ink); }
.game .dot.hit  { background: var(--real); border-color: var(--real); }
.game .dot.miss { background: transparent; border: clamp(3px, 0.9vw, 4px) solid var(--fake); }
.game.verdict { min-height: 118px; margin-top: 4px; }
.game.verdict p { font-size: 17px; line-height: 1.55; margin: 0; }
.game.verdict .truth { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
.game.verdict .truth.real { color: var(--real); }
.game.verdict .truth.fake { color: var(--fake); }
.game.verdict .hint { color: var(--muted); }
div[data-testid="stButton"] > button, div.stButton > button { width: 100%; min-height: 60px; }
div[data-testid="stButton"] > button p, div.stButton > button p { font-size: 18px; font-weight: 600; }
</style>
"""


def show(html, cls="game"):
    """Render a block of HTML. Lines are flattened so Markdown never treats indentation as code."""
    flat = " ".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(f'<div class="{cls}">{flat}</div>', unsafe_allow_html=True)


def label_text(y):
    return "a real photo" if y == 1 else "AI-generated"


def photo_html(g):
    if g["round"] >= N_ROUNDS:
        return final_html(g)
    idx = g["order"][g["round"]]
    stamp = ""
    if g["answered"]:                                         # stamp the answer onto the photo
        real = LABELS[idx] == 1
        stamp = f'<div class="stamp {"real" if real else "fake"}">{"REAL" if real else "AI"}</div>'
    return (f'<div class="print"><img src="{photo_uri(int(idx))}" '
            f'alt="Face {g["round"] + 1} of {N_ROUNDS}">{stamp}</div>')


def dots(results, current):
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


def board_html(g):
    finished = g["round"] >= N_ROUNDS
    current = -1 if finished else g["round"]
    you = [r["human_ok"] for r in g["history"]]
    bot = [r["model_ok"] for r in g["history"]]
    heading = "Final score" if finished else f"Face {g['round'] + 1} of {N_ROUNDS}"
    return f"""
    <p class="face-count">{heading}</p>
    <div class="track"><span class="who">You</span><span class="dots">{dots(you, current)}</span>
      <span class="num">{g['human']}</span></div>
    <div class="track"><span class="who">Model</span><span class="dots">{dots(bot, current)}</span>
      <span class="num">{g['model']}</span></div>
    """


def verdict_html(g):
    if g["round"] >= N_ROUNDS:
        return f'<p class="hint">Press Play again for {N_ROUNDS} new faces.</p>'
    if not g["answered"]:
        return '<p class="hint">Real photo or AI-generated? The model has already made its guess.</p>'
    last = g["history"][-1]
    truth_cls = "real" if last["truth"] == 1 else "fake"
    you = "You got it." if last["human_ok"] else "You missed this one."
    bot = (f"The model guessed {label_text(last['model_guess'])}, {last['model_conf']:.0%} sure, "
           f"and was {'right' if last['model_ok'] else 'wrong'}.")
    return f'<p class="truth {truth_cls}">It’s {label_text(last["truth"])}.</p><p>{you}</p><p>{bot}</p>'


def final_html(g):
    h, m = g["human"], g["model"]
    title = "You win" if h > m else ("The model wins" if m > h else "It’s a tie")
    return f"""
    <div class="final">
      <h2>{title}</h2>
      <div class="score">{h}<span class="to">to</span>{m}</div>
      <p>You spotted {h} of {N_ROUNDS} faces. The model spotted {m}.</p>
    </div>"""


# ---- Game logic (button callbacks run before the page redraws) ----
def new_game():
    st.session_state.game = {
        "order": random.sample(range(len(LABELS)), N_ROUNDS),   # different random faces each game
        "round": 0, "human": 0, "model": 0, "answered": False, "history": [],
    }


def make_guess(choice):
    g = st.session_state.game
    if g["answered"] or g["round"] >= N_ROUNDS:                  # ignore extra clicks
        return
    idx = g["order"][g["round"]]
    truth = int(LABELS[idx])
    model_guess = int(MODEL_PREDS[idx])
    p = float(MODEL_PROBS[idx])
    g["history"].append({
        "truth": truth,
        "human_ok": choice == truth,
        "model_ok": model_guess == truth,
        "model_guess": model_guess,
        "model_conf": p if model_guess == 1 else 1 - p,          # how sure the model was
    })
    g["human"] += int(choice == truth)
    g["model"] += int(model_guess == truth)
    g["answered"] = True


def next_round():
    g = st.session_state.game
    if g["round"] >= N_ROUNDS:                                   # after the last face: new game
        new_game()
    elif g["answered"]:
        g["round"] += 1
        g["answered"] = False


if "game" not in st.session_state:
    new_game()
g = st.session_state.game
finished = g["round"] >= N_ROUNDS
guessing = not g["answered"] and not finished
last_round = g["round"] == N_ROUNDS - 1

# ---- Layout ----
st.markdown(CSS, unsafe_allow_html=True)
show(f"""
<h1>Spot the fake</h1>
<p class="intro">Half of these faces are real photos and half were made by an AI. You and my {MODEL_LABEL} model
guess each one. After {N_ROUNDS} faces, whoever got more right wins.</p>
""")

left, right = st.columns([6, 5], gap="large")
with left:
    show(photo_html(g))
with right:
    show(board_html(g))
    c1, c2 = st.columns(2)
    c1.button("Real photo", key="real", on_click=make_guess, args=(1,), disabled=not guessing)
    c2.button("AI-generated", key="ai", on_click=make_guess, args=(0,), disabled=not guessing)
    show(verdict_html(g), cls="game verdict")
    next_label = "Play again" if finished else ("See who won" if last_round else "Next face")
    st.button(next_label, key="next", type="primary", on_click=next_round,
              disabled=not (g["answered"] or finished))
    st.button("Start a new game", key="restart", on_click=new_game)