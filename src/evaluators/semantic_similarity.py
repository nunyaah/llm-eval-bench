from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluators.base import BaseEvaluator


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Scores semantic similarity between expected and actual outputs using TF-IDF cosine similarity.

    Falls back to exact match for very short strings where TF-IDF has no vocabulary.
    """

    name = "semantic_similarity"

    def score(self, expected: str, actual: str) -> float:
        if not expected.strip() or not actual.strip():
            return 0.0

        # Fall back to exact match for short single-token strings (numbers, single words)
        # that would result in empty TF-IDF vocabulary after stop word filtering
        e = expected.strip().lower()
        a = actual.strip().lower()
        if len(e.split()) <= 1 and len(a.split()) <= 1:
            return 1.0 if e == a else 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([expected, actual])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except ValueError:
            # Empty vocabulary — fall back to exact match
            return 1.0 if e == a else 0.0
