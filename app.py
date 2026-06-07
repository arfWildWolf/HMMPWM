from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from Bio import Entrez, SeqIO
import io

# Import the class we just created in the previous step
from biogatekeeper import GatekeeperDeployer 

app = FastAPI(title="Gatekeeper TIS API")

# Allow the frontend GUI to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model engine once into memory
engine = GatekeeperDeployer(model_path='ensemble_model.pkl')
Entrez.email = "your.email@example.com" # Required for NCBI fetches

class PredictionResponse(BaseModel):
    sequence_length: int
    offset_start: int
    tis_positions: list[int]

def extract_tis_positions(state_path, offset=0):
    """Converts the [0,0,1,2,2] path array into exact coordinate numbers."""
    positions = []
    for i, state in enumerate(state_path):
        if state == 1:
            # i is the token index. Multiply by 3 for base pair index, add offset.
            positions.append((i * 3) + offset) 
    return positions

@app.post("/predict/string", response_model=PredictionResponse)
async def predict_string(sequence: str = Form(...), strategy: str = Form("v3")):
    if len(sequence) < 53:
        raise HTTPException(status_code=400, detail="Sequence too short.")
    
    path = engine.predict(sequence, strategy=strategy) # Dynamic strategy
    tis_coords = extract_tis_positions(path, offset=0)
    return {"sequence_length": len(sequence), "offset_start": 0, "tis_positions": tis_coords}

@app.post("/predict/file", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...), strategy: str = Form("v3")):
    content = await file.read()
    seq_str = ""
    try:
        text_content = content.decode("utf-8")
        if text_content.startswith(">"):
            fasta_io = io.StringIO(text_content)
            record = next(SeqIO.parse(fasta_io, "fasta"))
            seq_str = str(record.seq).upper()
        else:
            seq_str = text_content.replace("\n", "").replace(" ", "").upper()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format.")
        
    path = engine.predict(seq_str, strategy=strategy) # Dynamic strategy
    tis_coords = extract_tis_positions(path, offset=0)
    return {"sequence_length": len(seq_str), "offset_start": 0, "tis_positions": tis_coords}

@app.post("/predict/ncbi", response_model=PredictionResponse)
async def predict_ncbi(
    accession: str = Form(...), 
    start: int = Form(...), 
    end: int = Form(...),
    strategy: str = Form("v3")
):
    aligned_start = start - (start % 3)
    aligned_end = end + ((3 - (end % 3)) % 3)
    
    try:
        with Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text",
                           seq_start=aligned_start, seq_stop=aligned_end) as handle:
            record = SeqIO.read(handle, "fasta")
            seq_str = str(record.seq).upper()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NCBI Fetch failed: {str(e)}")

    path = engine.predict(seq_str, strategy=strategy) # Dynamic strategy
    absolute_offset = aligned_start - 1 
    tis_coords = extract_tis_positions(path, offset=absolute_offset)
    
    return {
        "sequence_length": len(seq_str), 
        "offset_start": absolute_offset, 
        "tis_positions": tis_coords
    }