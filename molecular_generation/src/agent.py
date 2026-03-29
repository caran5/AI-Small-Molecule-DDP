"""
Molecular Discovery Agent using Ollama (Free Local LLM)
Combines LogP prediction with LLM reasoning
"""

import json
import re
import subprocess
from typing import Optional, List, Dict
from src.predict import predict_logp


class OllamaAgent:
    """Agent that uses local Ollama LLM with LogP prediction tool"""
    
    def __init__(self, model="mistral"):
        """
        Initialize agent
        
        Args:
            model: Ollama model to use (mistral, llama2, neural-chat, etc.)
        """
        self.model = model
        self.conversation_history = []
        self.check_ollama()
    
    def check_ollama(self):
        """Check if Ollama is running"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print(f"✅ Ollama is installed: {result.stdout.strip()}")
            else:
                print("⚠️ Ollama command failed. Is it installed?")
                print("Install from: https://ollama.ai")
        except FileNotFoundError:
            print("❌ Ollama not found. Install from: https://ollama.ai")
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama LLM locally"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "⏱️ Response timeout - Ollama may be slow or offline"
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def extract_smiles(self, text: str) -> Optional[str]:
        """Extract SMILES from text using regex"""
        # Priority 1: Try "SMILES: ..." format (to end of line or sentence)
        smiles_colon = r'(?:SMILES|smiles)[\s:]+([A-Za-z0-9\[\]\(\)\\=\#\-\+@:/.,]+?)(?:\s*$|[\.\s,;])'
        match = re.search(smiles_colon, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Priority 2: Text after colon (captures full SMILES)
        colon_pattern = r':\s*([A-Za-z0-9\[\]\(\)\\=\#\-\+@:/.,]+?)(?:\s*$|[\.\s,;])'
        match = re.search(colon_pattern, text)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) > 3:
                return candidate
        
        # Priority 3: Text in parentheses (but be careful with nested parens)
        paren_pattern = r'\(([A-Za-z0-9\[\]\(\)\\=\#\-\+@:/.,]+)\)$'
        match = re.search(paren_pattern, text)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) > 3 and any(c in candidate for c in ['C', 'c', 'O', 'o', 'N', 'n']):
                return candidate
        
        return None
    
    def predict_and_explain(self, smiles: str) -> str:
        """Predict LogP and ask LLM to explain it"""
        # Get prediction
        result = predict_logp(smiles)
        
        if "error" in result:
            return f"❌ {result['error']}"
        
        # Ask LLM to interpret
        explanation_prompt = f"""
        A molecule has these properties:
        - SMILES: {smiles}
        - LogP (oiliness): {result['logp']}
        - Hydrophilicity: {result['hydrophilicity']}
        - Molecular Weight: {result['formula_weight']}
        - Hydrogen Donors: {result['h_donors']}
        - Hydrogen Acceptors: {result['h_acceptors']}
        - Rotatable Bonds: {result['rotatable_bonds']}
        
        In 2-3 sentences, explain:
        1. What does this LogP value mean?
        2. Is this good for a drug? Why or why not?
        3. What could be improved?
        """
        
        explanation = self._call_ollama(explanation_prompt)
        
        return f"""
📊 Prediction Result:
- LogP: {result['logp']}
- Classification: {result['hydrophilicity']}

🔬 Properties:
- Weight: {result['formula_weight']} g/mol
- H-Donors: {result['h_donors']} | H-Acceptors: {result['h_acceptors']}
- Rotatable Bonds: {result['rotatable_bonds']}

💡 AI Interpretation:
{explanation}
        """
    
    def chat(self, user_message: str) -> str:
        """
        Chat with the agent
        
        Handles:
        - Direct LogP predictions (if SMILES provided)
        - Questions about molecules
        - Molecular properties discussion
        """
        # Store in history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # First, always try to extract SMILES - regardless of keywords
        smiles = self.extract_smiles(user_message)
        
        if smiles:
            # Found SMILES - make a prediction!
            response = self.predict_and_explain(smiles)
        elif any(keyword in user_message.lower() for keyword in 
               ["logp", "predict", "smiles", "chemical", "molecule", "aspirin", 
                "ibuprofen", "acetaminophen", "painkiller", "drug"]):
            # Keywords suggest they want a prediction or molecule info
            # Ask LLM but focus on predictions
            enhanced_prompt = f"{user_message}\n\nIf you can identify a molecule SMILES, provide predictions."
            response = self._call_ollama(enhanced_prompt)
        else:
            # General question - ask LLM
            response = self._call_ollama(user_message)
        
        # Store response in history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def batch_compare(self, smiles_list: List[str]) -> str:
        """Compare multiple molecules"""
        predictions = [predict_logp(smi) for smi in smiles_list]
        
        # Format for LLM
        molecules_text = "\n".join([
            f"{i+1}. {p['smiles']}: LogP = {p['logp']} ({p['hydrophilicity']})"
            for i, p in enumerate(predictions)
        ])
        
        prompt = f"""
        Compare these molecules:
        {molecules_text}
        
        Which would be best for a drug? Why? Consider absorption, 
        distribution, and water solubility.
        """
        
        return self._call_ollama(prompt)
    
    def design_suggestion(self, requirements: str) -> str:
        """Ask LLM for molecule design suggestions"""
        prompt = f"""
        Design suggestions for a molecule with these requirements:
        {requirements}
        
        Consider:
        - LogP range (0-2 is usually good for drugs)
        - Molecular weight (< 500 for most drugs)
        - Number of hydrogen donors/acceptors (Lipinski's rule)
        
        Suggest general structure ideas (you don't need to give exact SMILES).
        """
        
        return self._call_ollama(prompt)
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def main():
    """Interactive agent CLI"""
    print("=" * 70)
    print("🧬 Molecular Discovery Agent (Free - Uses Local Ollama)")
    print("=" * 70)
    print("\nCommands:")
    print("  'predict SMILES' - Predict LogP for a molecule")
    print("  'compare SMILES1, SMILES2' - Compare multiple molecules")
    print("  'design: requirements' - Get design suggestions")
    print("  'history' - Show conversation history")
    print("  'clear' - Clear history")
    print("  'exit' - Quit")
    print("\nExample SMILES:")
    print("  Aspirin: CC(=O)Oc1ccccc1C(=O)O")
    print("  Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    print("-" * 70)
    
    agent = OllamaAgent()
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "exit":
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == "history":
                print("\n📝 Conversation History:")
                for msg in agent.get_history():
                    print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
                continue
            
            if user_input.lower() == "clear":
                agent.clear_history()
                print("✅ History cleared")
                continue
            
            if user_input.lower().startswith("design:"):
                requirements = user_input[7:].strip()
                print("\n🔬 AI Agent:")
                print(agent.design_suggestion(requirements))
            else:
                print("\n🔬 AI Agent:")
                print(agent.chat(user_input))
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
