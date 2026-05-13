from typing import List, Dict
import re
import json

# Latest Google GenAI library import
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


class ResponseEvaluator:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        if genai is None:
            raise RuntimeError("google-genai client is required. Run 'pip install google-genai'")
        
        if not api_key:
            # Pipeline-la irundhu pass aagura GOOGLE_API_KEY-ah check pandrom
            raise ValueError("API Key is required for Gemini evaluation")
            
        # Latest Client initialization
        self.client = genai.Client(api_key=api_key)
        self.model_id = model

    def assess(self, query: str, answer: str, sources: List[Dict]) -> Dict:
        source_text = "\n\n".join(
            [f"Source {idx + 1}: {item['text']}" for idx, item in enumerate(sources)]
        )
        
        prompt = (
            "You are a retrieval-augmented assistant reviewer. Evaluate the answer below for factual consistency "
            "against the provided document sources. Do not add information from outside the sources.\n\n"
            f"Query: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Sources:\n{source_text}\n\n"
            "Return a JSON object with strictly these keys: hallucination_score (0.0-1.0), consistency (high/medium/low), rationale (string). "
            "hallucination_score should be close to 0 if the answer is faithful, and close to 1 if it contains hallucinations."
        )

        try:
            # Using latest generation logic with JSON response hint
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            # Response-ah parse pandrom
            raw_data = json.loads(response.text)
            
            return {
                "raw_review": response.text,
                "hallucination_score": float(raw_data.get("hallucination_score", 0.5)),
                "consistency": str(raw_data.get("consistency", "medium")).lower(),
                "rationale": str(raw_data.get("rationale", "No rationale provided.")),
            }
            
        except Exception as e:
            # Error vandha fallback mechanism
            print(f"Evaluation error: {str(e)}")
            return {
                "raw_review": str(e),
                "hallucination_score": 0.5,
                "consistency": "medium",
                "rationale": f"Failed to parse evaluation: {str(e)}",
            }

    # Old regex methods are now secondary but kept for backward safety
    @staticmethod
    def _parse_score(raw: str) -> float:
        match = re.search(r"hallucination_score\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)", raw, re.IGNORECASE)
        if match:
            return min(max(float(match.group(1)), 0.0), 1.0)
        return 0.5

    @staticmethod
    def _parse_consistency(raw: str) -> str:
        low_indicators = ["inconsistent", "hallucination", "low"]
        high_indicators = ["consistent", "faithful", "high"]
        
        raw_lower = raw.lower()
        if any(word in raw_lower for word in low_indicators):
            return "low"
        if any(word in raw_lower for word in high_indicators):
            return "high"
        return "medium"