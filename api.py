"""
FastAPI Backend for BashAgent
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import time
from typing import Optional
 
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from config import (
    MODEL_SAVE_PATH, EMBEDDING_DIM, HIDDEN_DIM,
    NUM_LAYERS, DROPOUT, DEVICE
)
from inference import translate_command
from utils.entity_extractor import extract_entities, reinject_entities

# Initialize FastAPI app

app = FastAPI(
    title="BashAgent API",
    description="API for BashAgent - a command generation agent for Linux terminal tasks.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model State (loaded on startup)

class ModelState:
    model = None
    input_vocab = None
    output_vocab = None
    loaded = False
    load_time = None

state = ModelState()

@app.on_event("startup")
def load_model():
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
        state.input_vocab = checkpoint['input_vocab']
        state.output_vocab = checkpoint['output_vocab']
 
        encoder = Encoder(len(state.input_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
        decoder = Decoder(len(state.output_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
        state.model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
        state.model.load_state_dict(checkpoint['model_state_dict'])
        state.model.eval()
 
        state.loaded = True
        state.load_time = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        print("✅✅✅ BashAgent model loaded successfully.")
    except FileNotFoundError:
        print("⚠️⚠️⚠️  No checkpoint found. Train the model first with: python train.py")

# Request / Response schemas
class TranslateRequest(BaseModel):
    text: str
    debug: Optional[bool] = False # return intermediate pipeline steps
    model_type: Optional[str] = "t5" # lstm or "t5"

class TranslateResponse(BaseModel):
    input: str
    command: str
    debug_info: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    load_time: Optional[str] = None
    device: str
    input_vocab_size: Optional[int] = None
    output_vocab_size: Optional[int] = None


#endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check if the API and model are ready."""
    return HealthResponse(
        status="ok" if state.loaded else "model_not_loaded",
        model_loaded=state.loaded,
        load_time=state.load_time,
        device=str(DEVICE),
        input_vocab_size=len(state.input_vocab) if state.loaded else None,
        output_vocab_size=len(state.output_vocab) if state.loaded else None,
    )
 
 
@app.post("/translate", response_model=TranslateResponse, tags=["Inference"])
def translate(req: TranslateRequest):
    """
    Translate natural language to a bash command.
 
    Pipeline:
      1. Entity extraction  → replace named values with typed placeholders
      2. Seq2Seq inference  → produce template command
      3. Entity reinjection → substitute placeholders with real values
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
 
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
 
    # Step 1: Extract entities & canonicalize input
    canonical_input, entities = extract_entities(text)
 
    # Step 2: Run model inference on canonical form
    def model_fn(sentence: str) -> str:
        return translate_command(
            state.model, sentence, state.input_vocab, state.output_vocab, DEVICE
        )
 
    raw_command = model_fn(canonical_input)
 
    # Step 3: Reinject real entity values
    final_command = reinject_entities(raw_command, entities)
 
    debug_info = None
    if req.debug:
        debug_info = {
            "canonical_input": canonical_input,
            "raw_model_output": raw_command,
            "entities": {
                "name": entities.name,
                "extension": entities.extension,
                "filename": entities.filename,
                "directory": entities.directory,
                "number": entities.number,
            }
        }
 
    return TranslateResponse(
        input=text,
        command=final_command,
        debug_info=debug_info
    )
 
 
@app.get("/examples", tags=["Info"])
def get_examples():
    """Return example inputs to try in the UI."""
    return {
        "examples": [
            "list all files",
            "show current directory",
            "make a c file named bhagya",
            "create a python file called main",
            "remove file data.csv",
            "find all javascript files",
            "show last 20 lines of server.log",
            "make executable script named deploy.sh",
            "compress folder named myproject",
            "create a new directory called src",
        ]
    }