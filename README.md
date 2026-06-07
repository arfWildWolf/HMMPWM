# Gatekeeper: Explainable High-Resolution Cross-Eukaryotic TIS Predictor

Gatekeeper is a bioinformatics pipeline and interactive web application designed to predict Translation Initiation Sites (TIS) across diverse eukaryotic genomes. Utilizing a hybrid PWM (Position Weight Matrix) and HMM (Hidden Markov Model) architecture, it provides high-resolution sequence pattern recognition to identify exact gene start sites.

Unlike black-box models, Gatekeeper uses an explainable, tokenized codon-alignment strategy to score promoter regions (Kozak consensus) and downstream Codon Usage Bias (CUB). It includes a Decision Tree meta-router to automatically detect taxonomic clades and apply the correct sequence matrices.

## 🌟 Key Features
* **Dynamic Codon Alignment:** Automatically snaps raw NCBI coordinate queries to frame-zero boundaries.
* **Multi-Strategy Decoding:** Hot-swap between V3 (Contextual Log-Odds), HMM (Viterbi Tokenized), and Baseline prediction strategies.
* **Interactive Graphical UI:** A dark-themed local dashboard with an SVG visual track and glowing, score-based confidence markers.
* **Smart NCBI Caching:** Local `.pkl` caching prevents redundant API calls and speeds up model retraining.

---

## 📂 Project Structure

To run the program, ensure your project folder contains the following files:

1. `biogatekeeper.py` - The core machine learning backend, model logic, and training pipeline.
2. `app.py` - The FastAPI server that bridges the Python models to the web browser.
3. `index.html` - The graphical user interface.
4. `training_sources.json` - Your configured list of NCBI accessions used to train the model.
5. `testing_sources.json` - Your configured list of NCBI accessions used for benchmarking.

---

## 🚀 Installation & Setup (Beginner Guide)

### Step 1: Install Python & Prerequisites
Ensure you have Python 3.8 or newer installed on your system. Open your terminal (PowerShell or Command Prompt) and install the required scientific and server libraries:

```powershell
pip install fastapi uvicorn biopython pandas scikit-learn seaborn matplotlib python-multipart

### Step 2: First-Time Initialization (Training the Model)
Before the web interface can work, Gatekeeper must download the genetic data from NCBI, align the sequences, and compile the ensemble_model.pkl weight file.

Open your terminal in the project folder.

Run the core script:

PowerShell


python biogatekeeper.py
Wait for completion. The script will print its progress as it pulls sequences from NCBI, trains the clade-specific matrices, and trains the Decision Tree.

Once finished, you will see a new file appear in your folder named ensemble_model.pkl.

(Note: If you ever change the coordinates in your .json files, you must delete training_cache.pkl and testing_cache.pkl before running this script again to force a fresh download).

### Step 3: Start the Backend Server
Leave your terminal open and start the FastAPI server that powers the web interface:

PowerShell


uvicorn app:app --reload
(If PowerShell gives you an error saying uvicorn is not recognized, use this command instead: python -m uvicorn app:app --reload)

You should see a message saying: Application startup complete. Keep this terminal window open in the background.

### Step 4: Open the Graphical Interface
Navigate to your project folder in your file explorer.

Double-click the index.html file to open it in Google Chrome, Firefox, or Safari.

🖥️ How to Use the Application
Input Your Sequence:

NCBI Accession Tab: Type in an accession number (e.g., NM_001126112.2) and a coordinate range. The UI will automatically calculate the codon-aligned bounds.

Upload Tab: Drag and drop a .fasta or .txt sequence file.

Paste Tab: Manually paste a raw nucleotide string.

Select a Model Strategy:

Choose your preferred decoding algorithm from the dropdown menu in the bottom action bar.

Run Prediction:

Click Run Gatekeeper Prediction. The backend will process the sequence and return the structural path.

View Results:

Graphical View: The visual track will appear. TIS predictions are marked with glowing pins. Brighter/more opaque pins indicate a higher biological confidence score (PWM + CUB). Hover your mouse over a compressed notch to see the exact score.

List View: Click "View as List" to see a clean, scrollable readout of all predicted base-pair coordinates.

Export: Click "Export to CSV" to save the detected positions and their associated confidence scores to your local machine.

## ⚠️ Troubleshooting
Error: "Model not found at ensemble_model.pkl. Train first."
You forgot to run Step 2. Stop the server, run python biogatekeeper.py, and wait for it to finish before starting app.py.

Error: "NCBI Fetch failed" on the UI
NCBI servers might be rate-limiting you, or the Accession ID does not exist. Verify your internet connection and double-check the spelling of the Accession ID.

The SVG Track is empty (No red glowing pins)
The model analyzed the sequence but determined that no structural path met the minimum requirements for a true gene sequence (lacking Kozak consensus, poor CUB, or no ATGs present).