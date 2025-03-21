import pandas as pd
import os
import numpy as np
from typing import List, Dict, Any
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag.question_generators import simple_questions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

class DataProcessor:
    @staticmethod
    def create_knowledge_base(documents: List[str]) -> KnowledgeBase:
        """Create a knowledge base from document texts."""
        knowledge_base_df = pd.DataFrame(documents, columns=["text"])
        return KnowledgeBase.from_pandas(knowledge_base_df)

    @staticmethod
    def generate_test_questions(knowledge_base: KnowledgeBase, num_questions: int = 10):
        """Generate test questions from the knowledge base."""
        return generate_testset(
            knowledge_base,
            num_questions=num_questions,
            question_generators=[simple_questions],
            language='en',
            agent_description="A customer support chatbot for company X"
        )

    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text strings using TF-IDF."""
        if not text1 or not text2:
            return 0.0

        # Normalize text by removing punctuation and standardizing phone numbers
        def normalize_text(text):
            import re
            # Remove all punctuation except dots in phone numbers
            text = re.sub(r'[^\w\s.]', '', text)
            # Standardize phone numbers (remove dots)
            text = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1\2\3', text)
            return text.lower()

        text1_norm = normalize_text(text1)
        text2_norm = normalize_text(text2)

        # Check for exact match after normalization
        if text1_norm == text2_norm:
            return 1.0

        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3))
        try:
            tfidf_matrix = vectorizer.fit_transform([text1_norm, text2_norm])
            return float(sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0

    @staticmethod
    def calculate_context_metrics(generated: str, reference: str) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score for context evaluation."""
        if not generated or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        import re

        def normalize_text(text):
            # Remove all punctuation except dots in phone numbers
            text = re.sub(r'[^\w\s.]', '', text)
            # Standardize phone numbers (remove dots)
            text = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1\2\3', text)
            return text.lower()

        # Normalize both texts
        generated_norm = normalize_text(generated)
        reference_norm = normalize_text(reference)

        # Check for exact match after normalization
        if generated_norm == reference_norm:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        # Calculate metrics using character n-grams
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text)-n+1))

        generated_ngrams = get_ngrams(generated_norm)
        reference_ngrams = get_ngrams(reference_norm)
        
        common_ngrams = generated_ngrams.intersection(reference_ngrams)
        
        precision = len(common_ngrams) / len(generated_ngrams) if generated_ngrams else 0.0
        recall = len(common_ngrams) / len(reference_ngrams) if reference_ngrams else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def calculate_faithfulness(generated: str, reference: str) -> float:
        """Calculate faithfulness score based on semantic similarity and factual consistency."""
        if not generated or not reference:
            return 0.0

        # Combine exact match and semantic similarity
        similarity = DataProcessor.calculate_text_similarity(generated, reference)
        context_metrics = DataProcessor.calculate_context_metrics(generated, reference)
        
        # Weight both semantic and exact match components
        faithfulness = 0.6 * similarity + 0.4 * context_metrics["f1"]
        return faithfulness

    @staticmethod
    def process_metrics_data(metrics_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process metrics data into a pandas DataFrame with improved metric calculations."""
        df = pd.DataFrame(metrics_data)
        
        # Calculate metrics for each row
        for idx, row in df.iterrows():
            # Calculate cosine similarity
            df.at[idx, 'cosine_similarity'] = DataProcessor.calculate_text_similarity(
                row['generated_answer'], row['reference_context'])
            
            # Calculate context metrics
            context_metrics = DataProcessor.calculate_context_metrics(
                row['generated_answer'], row['reference_context'])
            df.at[idx, 'context_precision'] = context_metrics['precision']
            df.at[idx, 'context_recall'] = context_metrics['recall']
            df.at[idx, 'context_f1'] = context_metrics['f1']
            
            # Calculate faithfulness
            df.at[idx, 'faithfulness'] = DataProcessor.calculate_faithfulness(
                row['generated_answer'], row['reference_context'])
        
        return df

    @staticmethod
    def calculate_average_metrics(metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate average values for each metric."""
        metric_columns = [
            'cosine_similarity',
            'context_precision',
            'context_recall',
            'context_f1',
            'faithfulness'
        ]
        return {col: metrics_df[col].mean() for col in metric_columns if col in metrics_df.columns}

    @staticmethod
    def serialize_testset(testset):
        from datetime import datetime
        output = []
        for q in testset:
            question = q.question
            reference_context = getattr(q, 'context', '')
            reference_answer = q.reference_answer
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'difficulty_level': 'medium',
                'question_type': 'simple',
                'language': 'en',
                'source_type': 'document',
                'version': '1.0'
            }
            print(f"Question: {question}")
            print(f"Reference Context: {reference_context}")
            print(f"Reference Answer: {reference_answer}")
            print(f"Metadata: {metadata}\n")
            
            output.append({
                'question': question,
                'reference_context': reference_context,
                'reference_answer': reference_answer,
                'metadata': metadata
            })
        return output

    @staticmethod
    def save_testset(testset: Dict[str, Any], file_path: str) -> None:
        """Save serialized test set to a JSONL file with error handling."""
        try:
            import json
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                for item in testset:
                    f.write(json.dumps(item) + '\n')
        except Exception as e:
            raise Exception(f"Error saving test set: {str(e)}")

    @staticmethod
    def load_documents(file_path: str) -> List[str]:
        """Load documents from a file with proper error handling."""
        if not isinstance(file_path, str):
            raise TypeError(f"Expected string file path, got {type(file_path)}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")