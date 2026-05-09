# BashAgent 🤖

A natural language to bash command translator built from scratch using a Seq2Seq LSTM architecture with Bahdanau attention. You describe what you want to do in plain English — the model outputs the correct bash command.

Built by **Bhagya Pathiranage** — Software Engineering student at Carleton University.

---

## Why I Built This

I wanted a hands-on project that forced me to implement a full NLP pipeline end-to-end — not just call an API. The goal was to understand how sequence-to-sequence models actually work: how attention helps the decoder focus on relevant input tokens, how teacher forcing stabilizes training, and what it takes to go from raw training data to a deployed inference endpoint.

I also wanted to solve a real problem: I regularly forget obscure bash syntax. Instead of Googling `tar` flags for the tenth time, I wanted to just type what I mean.

---

## Architecture

```
User Input (natural language)
        │
        ▼
┌─────────────────────┐
│  Entity Extractor   │  ← extracts names, extensions, numbers
│  entity_extractor.py│     before the model sees the sentence
└─────────────────────┘
        │ canonical sentence (with <NAME>, <EXT>, <FILENAME> slots)
        ▼
┌─────────────────────────────────────────────────┐
│                  Seq2Seq Model                  │
│                                                 │
│  Encoder (2-layer LSTM)                         │
│    → encodes input tokens into hidden states    │
│                                                 │
│  Bahdanau Attention                             │
│    → decoder learns which input tokens matter   │
│       at each output step                       │
│                                                 │
│  Decoder (2-layer LSTM)                         │
│    → autoregressively generates output tokens   │
│    → teacher forcing during training            │
└─────────────────────────────────────────────────┘
        │ template command (e.g. "touch <NAME>.<EXT>")
        ▼
┌─────────────────────┐
│  Entity Reinjection │  ← substitutes real values back in
└─────────────────────┘
        │
        ▼
  bash command (e.g. "touch bhagya.c")
```

The key insight: the seq2seq model only needs to learn *patterns*, not specific filenames or directory names. The entity extractor handles all the dynamic values, which means the model generalizes cleanly to names it has never seen.

---

## Project Structure

```
bash-agent-pytorch/
├── api.py                  # FastAPI backend — /translate, /health, /examples
├── train.py                # Training loop with teacher forcing
├── inference.py            # Single-sentence inference
├── interactive.py          # CLI interface (terminal chat)
├── evaluate.py             # Accuracy reporting
├── test.py                 # Quick interactive test
├── test_massive_dataset.py # Full benchmark suite
├── config.py               # Hyperparameters and paths
├── start.ps1               # Auto-launcher (Windows)
├── start.sh                # Auto-launcher (Linux/macOS)
│
├── data/
│   ├── dataset.py          # PyTorch Dataset + collate_fn
│   └── raw_commands.json   # ~385 training pairs (flat + placeholder variants)
│
├── models/
│   ├── encoder.py          # LSTM encoder with packed sequences
│   ├── decoder.py          # LSTM decoder with attention
│   ├── attention.py        # Bahdanau attention mechanism
│   └── seq2seq.py          # Seq2Seq wrapper
│
├── utils/
│   ├── vocabulary.py       # word↔index mapping
│   ├── train_utils.py      # epoch loop + checkpoint saving
│   ├── evaluation.py       # exact match + token-level accuracy
│   └── entity_extractor.py # NLP preprocessing pipeline
│
├── checkpoints/
│   └── bash_agent_best.pth # saved model (best loss checkpoint)
│
└── NEXTJS-UI/              # Terminal-style frontend
    └── app/
        ├── page.jsx
        └── page.module.css
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+ (for the UI)

### Option 1 — Auto Launcher (recommended)

**Windows:**
```powershell
.\start.ps1
```

**Linux / macOS:**
```bash
chmod +x start.sh
./start.sh
```

The script will:
1. Create a virtual environment if one doesn't exist
2. Install all Python dependencies
3. **Train the model automatically** if no checkpoint is found
4. Start the FastAPI backend on `localhost:8000`
5. Start the Next.js frontend on `localhost:3000`

### Option 2 — Manual

```bash
# 1. Set up environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Train the model
python train.py

# 4. Start backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 5. Start frontend (separate terminal)
cd NEXTJS-UI && npm install && npm run dev
```

---

## Training

The model trains on ~385 input/output pairs covering file operations, process management, networking, permissions, package management, and system info — with both concrete examples (e.g. `remove file data.csv`) and parameterized templates (e.g. `remove file <FILENAME>`).

**Config (`config.py`):**
| Parameter | Value | Notes |
|---|---|---|
| `EMBEDDING_DIM` | 256 | token embedding size |
| `HIDDEN_DIM` | 512 | LSTM hidden state size |
| `NUM_LAYERS` | 3 | stacked LSTM layers |
| `DROPOUT` | 0.3 | applied between LSTM layers |
| `BATCH_SIZE` | 32 | |
| `LEARNING_RATE` | 0.0005 | Adam optimizer |
| `NUM_EPOCHS` | 500 | |
| `TEACHER_FORCING_RATIO` | 0.5 | probability of using ground truth token |

---

## API Reference

Once the backend is running, interactive docs are at `http://localhost:8000/docs`.

### `POST /translate`
```json
{
  "text": "make a c file named bhagya",
  "debug": true
}
```
**Response:**
```json
{
  "input": "make a c file named bhagya",
  "command": "touch bhagya.c",
  "debug_info": {
    "canonical_input": "make a <EXT> file named <NAME>",
    "raw_model_output": "touch <NAME>.<EXT>",
    "entities": { "name": "bhagya", "extension": "c" }
  }
}
```

### `GET /health`
Returns model status, vocab sizes, and device.

### `GET /examples`
Returns example inputs to try.

---

## How the Entity Extractor Works

The seq2seq model can only produce tokens it saw during training. Filenames like `bhagya` or `server.log` were never in training — so instead of hallucinating, the pipeline extracts them before inference and reinserts them after:

```
"show last 20 lines of server.log"
  → extract: name=server, ext=log, number=20
  → canonical: "show last <NUM> lines of <FILENAME>"
  → model output: "tail -n <NUM> <FILENAME>"
  → reinject: "tail -n 20 server.log"
```

---

## What I Learned

- **Seq2Seq with attention from scratch** — implementing Bahdanau attention helped me understand why transformers work: the attention mechanism is what lets the decoder know *where to look* in the input at each generation step.
- **Teacher forcing trade-offs** — too high and the model can't recover at inference time; too low and training is unstable early on. 0.5 was a solid middle ground.
- **Vocabulary generalization** — the fundamental limitation of seq2seq: it can only emit tokens it saw in training. The entity extraction pipeline is a practical solution to this without needing a much larger model.
- **FastAPI + PyTorch serving** — loading a model once at startup via `@app.on_event("startup")` and sharing state is the right pattern for inference APIs.

---

## Future Improvements

- [ ] Beam search decoding (currently greedy)
- [ ] Multi-command outputs (e.g. `mkdir src && cd src`)
- [ ] Confidence scores on predictions
- [ ] Fine-tune on user feedback (thumbs up / down in the UI)
- [ ] Replace LSTM backbone with a small transformer encoder

---

## License

MIT — feel free to use, modify, and build on this.