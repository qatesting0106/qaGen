import os
import json
from datetime import datetime
from typing import Dict, Any

class FileHandler:
    def __init__(self, output_dir: str):
        """Initialize FileHandler with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self, data: Dict[str, Any], prefix: str = "data") -> str:
        """Save data to a JSON file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        file_path = os.path.join(self.output_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        return file_path

    def save_jsonl(self, data: Dict[str, Any], prefix: str = "data") -> str:
        """Save data to a JSONL file with timestamp and validation."""
        from ..core.security.security_assessment import SecurityAssessment
        
        # Validate and enhance security evaluation structure
        if 'security_evaluation' in data:
            sec_assessment = SecurityAssessment()
            categories = data['security_evaluation'].get('categories', {})
            
            # Ensure all LLM categories exist with proper structure
            for cat_id in sec_assessment.category_descriptions:
                if cat_id not in categories:
                    categories[cat_id] = {
                        'name': sec_assessment.category_descriptions[cat_id],
                        'vulnerability_count': 0,
                        'risk_level': 'low',
                        'detected_payloads': []
                    }
            
            # Calculate missing risk levels using security assessment weights
            for cat_id, details in categories.items():
                if 'risk_level' not in details or details['risk_level'] == 'UNKNOWN':
                    risk_score = details.get('risk_score', sec_assessment.risk_weights.get(cat_id, 0.5))
                    details['risk_level'] = 'critical' if risk_score >= 0.8 else \
                                           'high' if risk_score >= 0.6 else \
                                           'medium' if risk_score >= 0.4 else 'low'
        
        # Validate complete JSON schema
        schema = {
            "type": "object",
            "required": ["security_evaluation"],
            "properties": {
                "security_evaluation": {
                    "type": "object",
                    "required": ["categories"],
                    "properties": {
                        "categories": {
                            "type": "object",
                            "minProperties": 13,
                            "patternProperties": {
                                "^LLM\\d{2}$": {
                                    "type": "object",
                                    "required": ["name", "vulnerability_count", "risk_level"]
                                }
                            }
                        }
                    }
                }
            }
        }
        from jsonschema import validate
        validate(instance=data, schema=schema)

        # Proceed with saving validated data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jsonl"
        file_path = os.path.join(self.output_dir, filename)

        with open(file_path, 'a') as f:
            json.dump(data, f)
            f.write('\n')

        return file_path

    def get_latest_file(self, extension: str) -> str:
        """Get the latest file with specified extension from output directory."""
        files = [f for f in os.listdir(self.output_dir) if f.endswith(extension)]
        if not files:
            return None
        return os.path.join(self.output_dir, max(files, key=lambda x: os.path.getctime(os.path.join(self.output_dir, x))))

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load data from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            return json.load(f)

    def load_jsonl(self, file_path: str) -> list:
        """Load data from a JSONL file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data