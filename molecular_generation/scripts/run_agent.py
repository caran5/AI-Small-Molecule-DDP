#!/usr/bin/env python3
"""
Quick start script for Molecular Discovery Agent
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import OllamaAgent, main

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        agent = OllamaAgent()
        query = " ".join(sys.argv[1:])
        print(agent.chat(query))
    else:
        # Interactive mode
        main()
