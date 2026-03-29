# Molecular Discovery Agent Setup

## ✅ Quick Start

### 1. Install Ollama (One-time Setup)

**macOS:**
```bash
brew install ollama
# Or download from https://ollama.ai/download/mac
```

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download/windows

### 2. Start Ollama

```bash
ollama serve
```

This starts the Ollama server on your machine. Keep this running in a terminal.

### 3. Pull a Model (One-time, ~4GB download)

In another terminal:

```bash
# Mistral (Recommended - fast & smart)
ollama pull mistral

# Or alternatives:
ollama pull llama2          # More capable but slower
ollama pull neural-chat     # Balanced
```

### 4. Run the Agent

```bash
cd /Users/ceejayarana/diffusion_model/molecular_generation
python scripts/run_agent.py
```

## 📖 Usage Examples

### Interactive Mode

```bash
python scripts/run_agent.py
```

Then chat:
```
🤖 You: predict SMILES: CC(=O)Oc1ccccc1C(=O)O

🔬 AI Agent:
📊 Prediction Result:
- LogP: 1.19
- Classification: Balanced (good absorption)

[LLM explains what this means]
```

### Single Query Mode

```bash
python scripts/run_agent.py "What is LogP and why does it matter?"
python scripts/run_agent.py "predict SMILES: CC(=O)Oc1ccccc1C(=O)O"
python scripts/run_agent.py "compare SMILES: CCO, CC(=O)O, CC(=O)Oc1ccccc1C(=O)O"
```

## 🧪 Test Molecules

Try these SMILES:

| Molecule | SMILES |
|----------|--------|
| **Aspirin** | `CC(=O)Oc1ccccc1C(=O)O` |
| **Caffeine** | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` |
| **Ethanol** | `CCO` |
| **Acetic Acid** | `CC(=O)O` |
| **Ibuprofen** | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |

## 📊 What You Can Do

1. **Predict LogP**
   - Input: SMILES string
   - Output: LogP value + interpretation + drug suitability

2. **Compare Molecules**
   - Input: Multiple SMILES
   - Output: Which is best, why

3. **Get Design Suggestions**
   - Input: Requirements (e.g., "lipophilic, small, low toxicity")
   - Output: Ideas for molecular structures

4. **Ask Questions**
   - Input: Any question about molecules, drugs, chemistry
   - Output: AI-generated explanation

## ⚙️ Configuration

### Change Model

Edit `scripts/run_agent.py`:
```python
agent = OllamaAgent(model="llama2")  # or "neural-chat", etc.
```

### Model Speeds & Smarts

| Model | Speed | Intelligence | Memory |
|-------|-------|--------------|--------|
| mistral | ⚡ Fast | 🧠 Very Good | 7B |
| llama2 | 🐢 Slow | 🧠🧠 Excellent | 7B |
| neural-chat | ⚡ Fast | 🧠 Good | 7B |
| dolphin-mixtral | 🐢 Slow | 🧠🧠 Excellent | 46B |

## 🚀 Next: Advanced Version

Once you're comfortable with this, we can:

1. **Add More Tools**
   - Lipinski's rule checker
   - Molecular weight calculator
   - Bioavailability predictor

2. **Connect to Claude API (Optional)**
   - Use Claude for complex reasoning
   - Keep Ollama for simple predictions
   - Hybrid approach = best of both

3. **Build Web Interface**
   - Flask/FastAPI server
   - Web UI for predictions
   - Share with others

## ❓ Troubleshooting

### "Ollama not found"
- Make sure `ollama serve` is running in another terminal
- Check installation: `ollama --version`

### "Model not found"
- Pull model first: `ollama pull mistral`

### "Very slow responses"
- Ollama runs on your CPU by default
- GPU support available for faster responses
- Try smaller model: `mistral` instead of `llama2`

### "Memory error"
- Close other applications
- Use smaller model
- Ollama requires ~8GB RAM minimum

## 📝 File Structure

```
src/
├── predict.py       ← LogP prediction function
└── agent.py        ← Ollama agent

scripts/
└── run_agent.py    ← CLI entry point
```

## 💡 Tips

1. **First run is slow** - Model loads into memory (~5GB)
2. **GPU helps** - If you have NVIDIA/AMD GPU, it's much faster
3. **Leave Ollama running** - No need to restart between queries
4. **Try different models** - Each has different strengths

Enjoy! 🧬🤖
