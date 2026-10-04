import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
import numpy as np
import os

class ModelEvaluator:
    """
    Evaluates and compares different AI models based on:
    - Latency (speed)
    - Question quality (length, coherence)
    """
    
    def __init__(self):
        self.results = []
        self.load_existing_results()
        
    def load_existing_results(self):
        """Load previous evaluation results if they exist"""
        try:
            if os.path.exists('evaluation_results.json'):
                with open('evaluation_results.json', 'r') as f:
                    self.results = json.load(f)
        except Exception:
            self.results = []
            
    def evaluate_model(self, model_name: str, questions_data: List[Dict]) -> Dict:
        if not questions_data:
            return {}
            
        latencies = [q.get('latency', 0) for q in questions_data]
        question_lengths = [len(q.get('content', '')) for q in questions_data]
        
        metrics = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'avg_latency': np.mean(latencies) if latencies else 0,
            'std_latency': np.std(latencies) if latencies else 0,
            'total_questions': len(questions_data),
            'avg_question_length': np.mean(question_lengths) if question_lengths else 0,
            # Simple heuristic score: Balance between speed (lower latency) and detail (length)
            'quality_score': (1.0 / (np.mean(latencies) + 0.1)) * (np.mean(question_lengths) / 100)
        }
        
        self.results.append(metrics)
        return metrics

    def compare_models(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).sort_values('timestamp', ascending=False)
        
    def save_results(self, filename: str = "evaluation_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)