import requests
from typing import List, Dict
import subprocess

class ModelManager:
    def __init__(self):
        self.base_url = "http://localhost:11434"

    def get_installed_models(self) -> List[str]:
        """Get list of installed Ollama models"""
        try:
            response = requests.get(f'{self.base_url}/api/tags')
            if response.status_code == 200:
                models = response.json()['models']
                return [model['name'] for model in models]
            return []
        except Exception:
            return []

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f'{self.base_url}/api/tags')
            return response.status_code == 200
        except Exception:
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a new Ollama model"""
        try:
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=True, 
                                  text=True)
            return result.returncode == 0
        except Exception:
            return False

    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        try:
            response = requests.post(
                f'{self.base_url}/api/show',
                json={'name': model_name}
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}