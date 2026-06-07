from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from Bio import Entrez, SeqIO
import io

# Import the class we created previously
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

# --- NEW: Updated Response Models ---
class TISDetail(BaseModel):
    pos: int
    score: float

class PredictionResponse(BaseModel):
    sequence_length: int
    offset_start: int
    tis_details: List[TISDetail] # Replaced tis_positions

def get_tis_details(state_path, seq_str, params, offset=0):
    """
    Converts the [0,0,1,2,2] path array into exact coordinate numbers 
    and calculates a normalized confidence score for the UI glow effect.
    """
    details = []
    for i, state in enumerate(state_path):
        if state == 1:
            idx = i * 3
            pos = idx + offset
            score = 1.0 # Default fallback
            
            # Re-calculate the contextual score to power the visual glow opacity
            try:
                upstream = seq_str[idx-50:idx]
                downstream = seq_str[idx+3:idx+93]
                
                if len(upstream) == 50 and len(downstream) >= 90:
                    p_score = engine._score_promoter(upstream, params['pwm_logo'])
                    cub_score = engine._score_cub(downstream, params['codon_log_odds'])
                    
                    raw_score = p_score + (cub_score * 2.0)
                    
                    # Normalize raw biological score to a 0.3 - 1.0 range for CSS opacity
                    # (Assuming raw_score typically falls between -10 and +10)
                    normalized = (raw_score + 10) / 20.0
                    score = max(0.3, min(1.0, normalized))
            except Exception:
                pass # Keep default 1.0 if sequence boundaries are too tight
                
            details.append({"pos": pos, "score": score})
            
    return details

@app.post("/predict/string", response_model=PredictionResponse)
async def predict_string(sequence: str = Form(...), strategy: str = Form("v3")):
    if len(sequence) < 53:
        raise HTTPException(status_code=400, detail="Sequence too short.")
    
    path = engine.predict(sequence, strategy=strategy)
    tis_details = get_tis_details(path, sequence, engine.default_params, offset=0)
    
    return {
        "sequence_length": len(sequence), 
        "offset_start": 0, 
        "tis_details": tis_details
    }

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
        
    path = engine.predict(seq_str, strategy=strategy)
    tis_details = get_tis_details(path, seq_str, engine.default_params, offset=0)
    
    return {
        "sequence_length": len(seq_str), 
        "offset_start": 0, 
        "tis_details": tis_details
    }

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

    path = engine.predict(seq_str, strategy=strategy)
    absolute_offset = aligned_start - 1 
    tis_details = get_tis_details(path, seq_str, engine.default_params, offset=absolute_offset)
    
    return {
        "sequence_length": len(seq_str), 
        "offset_start": absolute_offset, 
        "tis_details": tis_details
    }