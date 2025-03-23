import os
import json
import numpy as np
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_mistralai.embeddings import MistralAIEmbeddings
import asyncio


class RagasEmbeddingsWrapper(MistralAIEmbeddings):
    async def aembed_documents(self, texts, **kwargs):
        return await asyncio.to_thread(self.embed_documents, texts, **kwargs)

    async def aembed_query(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_query, text, **kwargs)


class MetricsEvaluator:
    def __init__(self, output_dir, embeddings):
        self.output_dir = os.path.abspath(output_dir)
        self.embeddings = embeddings
        os.makedirs(output_dir, exist_ok=True)

    def calculate_metrics(self, generated_answer, reference_context, question):
        """Calculate various metrics for the generated answer."""
        # Calculate base metrics
        base_metrics = {
            'question': question,
            'generated_answer': generated_answer,
            'reference_context': reference_context,
            'cosine_similarity': self._calculate_cosine_similarity(generated_answer, reference_context),
            'context_precision': self._calculate_context_precision(generated_answer, reference_context),
            'context_recall': self._calculate_context_recall(generated_answer, reference_context),
            'context_f1': self._calculate_context_f1(generated_answer, reference_context),
            'faithfulness': self._calculate_faithfulness(generated_answer, reference_context),
            'answer_relevance': self._calculate_answer_relevance(generated_answer, question),
            'answer_completeness': self._calculate_answer_completeness(generated_answer, reference_context),
            'answer_consistency': self._calculate_answer_consistency(generated_answer, reference_context)
        }
        
        return base_metrics

    def _calculate_cosine_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _calculate_context_precision(self, generated_answer, reference_context):
        """Calculate precision of generated answer against reference context."""
        try:
            generated_tokens = set(self._tokenize(generated_answer))
            reference_tokens = set(self._tokenize(reference_context))
            if not generated_tokens:
                return 0.0
            return len(generated_tokens.intersection(reference_tokens)) / len(generated_tokens)
        except Exception as e:
            print(f"Error calculating context precision: {e}")
            return 0.0

    def _calculate_context_recall(self, generated_answer, reference_context):
        """Calculate recall of generated answer against reference context."""
        try:
            generated_tokens = set(self._tokenize(generated_answer))
            reference_tokens = set(self._tokenize(reference_context))
            if not reference_tokens:
                return 0.0
            return len(generated_tokens.intersection(reference_tokens)) / len(reference_tokens)
        except Exception as e:
            print(f"Error calculating context recall: {e}")
            return 0.0

    def _calculate_context_f1(self, generated_answer, reference_context):
        """Calculate F1 score between generated answer and reference context."""
        try:
            precision = self._calculate_context_precision(generated_answer, reference_context)
            recall = self._calculate_context_recall(generated_answer, reference_context)
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        except Exception as e:
            print(f"Error calculating context F1: {e}")
            return 0.0

    def _calculate_faithfulness(self, generated_answer, reference_context):
        """Calculate faithfulness score of generated answer to reference context."""
        try:
            # Use embeddings to calculate semantic similarity
            answer_embedding = self.embeddings.embed_query(generated_answer)
            context_embedding = self.embeddings.embed_query(reference_context)
            similarity = cosine_similarity([answer_embedding], [context_embedding])[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Error calculating faithfulness: {e}")
            return 0.0

    def _calculate_answer_relevance(self, generated_answer, question):
        """Calculate relevance of generated answer to the question."""
        try:
            # Use embeddings to calculate semantic similarity
            answer_embedding = self.embeddings.embed_query(generated_answer)
            question_embedding = self.embeddings.embed_query(question)
            similarity = cosine_similarity([answer_embedding], [question_embedding])[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Error calculating answer relevance: {e}")
            return 0.0

    def _calculate_answer_completeness(self, generated_answer, reference_context):
        """Calculate completeness of generated answer relative to reference context."""
        try:
            # Simple length-based completeness score
            answer_length = len(self._tokenize(generated_answer))
            context_length = len(self._tokenize(reference_context))
            if context_length == 0:
                return 0.0
            completeness = answer_length / context_length
            return max(0.0, min(1.0, completeness))
        except Exception as e:
            print(f"Error calculating answer completeness: {e}")
            return 0.0

    def _calculate_answer_consistency(self, generated_answer, reference_context):
        """Calculate consistency between generated answer and reference context."""
        try:
            # Use embeddings to calculate semantic consistency
            answer_embedding = self.embeddings.embed_query(generated_answer)
            context_embedding = self.embeddings.embed_query(reference_context)
            consistency = cosine_similarity([answer_embedding], [context_embedding])[0][0]
            return max(0.0, min(1.0, consistency))
        except Exception as e:
            print(f"Error calculating answer consistency: {e}")
            return 0.0

    def _tokenize(self, text):
        """Simple tokenization function."""
        return re.findall(r'\w+', text.lower())