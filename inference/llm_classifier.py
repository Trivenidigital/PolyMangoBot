"""
LLM Classifier Module
=====================

Uses LLM (Claude/GPT) to classify market relationships when
token-based matching is insufficient or ambiguous.

Provides fallback classification for complex market families.
"""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from inference.models import (
    PolymarketMarket,
    RelationshipType,
)

logger = logging.getLogger("PolyMangoBot.inference.llm_classifier")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM classifier"""
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-3-haiku-20240307"  # Fast and cheap for classification
    api_key: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.0  # Deterministic for classification
    timeout_seconds: float = 10.0
    max_retries: int = 2
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour cache


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

RELATIONSHIP_CLASSIFICATION_PROMPT = """Analyze these prediction market questions and classify their logical relationship.

Questions:
{questions}

Classify the relationship as ONE of:
- "date_variant": Nested deadlines for the same event (e.g., "by March" vs "by April")
- "exclusive": At most one can be YES (mutually exclusive outcomes)
- "exhaustive": At least one must be YES (exhaustive set)
- "exact": Exactly one will be YES (exclusive AND exhaustive)
- "conditional": One implies another (A implies B)
- "unknown": Cannot determine relationship

Also identify:
1. The shared entity/subject (e.g., "Bitcoin", "Trump", "Lakers")
2. The shared event/predicate (e.g., "reach price", "win election", "make playoffs")
3. Confidence score (0.0-1.0)

Respond in JSON format:
{{
    "relationship": "<type>",
    "shared_entity": "<entity or null>",
    "shared_event": "<event or null>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}"""


FAMILY_GROUPING_PROMPT = """Given these prediction market questions, identify which ones belong together as a logical family (related questions about the same underlying event).

Questions:
{questions}

Group related questions by their shared subject. Return JSON:
{{
    "groups": [
        {{
            "indices": [0, 1, 3],
            "shared_subject": "description of what links them",
            "confidence": 0.85
        }}
    ],
    "ungrouped": [2, 4]
}}"""


# =============================================================================
# CACHE
# =============================================================================

class LLMCache:
    """Simple in-memory cache for LLM responses"""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def _hash_key(self, prompt: str) -> str:
        # MD5 used for cache keying only, not security
        return hashlib.md5(prompt.encode(), usedforsecurity=False).hexdigest()

    def get(self, prompt: str) -> Optional[Any]:
        import time
        key = self._hash_key(prompt)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, prompt: str, value: Any) -> None:
        import time
        key = self._hash_key(prompt)
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        self._cache.clear()


# =============================================================================
# LLM PROVIDER INTERFACE
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def complete(self, prompt: str, config: LLMConfig) -> str:
        """Send prompt to LLM and return response"""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    async def complete(self, prompt: str, config: LLMConfig) -> str:
        try:
            import anthropic
        except ImportError as err:
            raise ImportError("anthropic package required: pip install anthropic") from err

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)

        message = await client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text from the first content block
        first_content = message.content[0]
        if hasattr(first_content, 'text'):
            return first_content.text  # type: ignore[union-attr]
        return str(first_content)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    async def complete(self, prompt: str, config: LLMConfig) -> str:
        try:
            import openai
        except ImportError as err:
            raise ImportError("openai package required: pip install openai") from err

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = openai.AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=config.model if "gpt" in config.model else "gpt-4o-mini",
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        return content or ""


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls"""

    def __init__(self, responses: Optional[dict[str, str]] = None):
        self._responses = responses or {}
        self._default_response = json.dumps({
            "relationship": "unknown",
            "shared_entity": None,
            "shared_event": None,
            "confidence": 0.5,
            "reasoning": "Mock response"
        })

    async def complete(self, prompt: str, config: LLMConfig) -> str:
        # Check for keyword-based responses
        for keyword, response in self._responses.items():
            if keyword.lower() in prompt.lower():
                return response
        return self._default_response


# =============================================================================
# LLM CLASSIFIER
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of LLM classification"""
    relationship: RelationshipType
    shared_entity: Optional[str]
    shared_event: Optional[str]
    confidence: float
    reasoning: str
    from_cache: bool = False


class LLMClassifier:
    """
    Uses LLM to classify market relationships.

    Provides fallback classification when token-based matching
    is insufficient or ambiguous.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._cache = LLMCache(self.config.cache_ttl_seconds)
        self._provider = self._create_provider()
        self._call_count = 0
        self._cache_hits = 0

    def _create_provider(self) -> LLMProvider:
        """Create LLM provider based on config"""
        if self.config.provider == "anthropic":
            return AnthropicProvider()
        elif self.config.provider == "openai":
            return OpenAIProvider()
        elif self.config.provider == "mock":
            return MockProvider()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    async def classify_relationship(
        self,
        markets: list[PolymarketMarket]
    ) -> ClassificationResult:
        """
        Classify the relationship between markets using LLM.

        Args:
            markets: List of markets to classify

        Returns:
            ClassificationResult with relationship type and metadata
        """
        if len(markets) < 2:
            return ClassificationResult(
                relationship=RelationshipType.UNKNOWN,
                shared_entity=None,
                shared_event=None,
                confidence=0.0,
                reasoning="Need at least 2 markets to classify"
            )

        # Build prompt
        questions = "\n".join(
            f"{i+1}. {m.question}"
            for i, m in enumerate(markets)
        )
        prompt = RELATIONSHIP_CLASSIFICATION_PROMPT.format(questions=questions)

        # Check cache
        if self.config.cache_enabled:
            cached = self._cache.get(prompt)
            if cached is not None:
                self._cache_hits += 1
                result: ClassificationResult = cached
                result.from_cache = True
                return result

        # Call LLM
        self._call_count += 1

        try:
            response = await self._provider.complete(prompt, self.config)
            result = self._parse_classification_response(response)

            # Cache result
            if self.config.cache_enabled:
                self._cache.set(prompt, result)

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return ClassificationResult(
                relationship=RelationshipType.UNKNOWN,
                shared_entity=None,
                shared_event=None,
                confidence=0.0,
                reasoning=f"Error: {e!s}"
            )

    def _parse_classification_response(self, response: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            # Map relationship string to enum
            rel_map = {
                "date_variant": RelationshipType.DATE_VARIANT,
                "exclusive": RelationshipType.MUTUALLY_EXCLUSIVE,
                "exhaustive": RelationshipType.EXHAUSTIVE,
                "exact": RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE,
                "conditional": RelationshipType.CONDITIONAL,
                "unknown": RelationshipType.UNKNOWN,
            }

            relationship = rel_map.get(
                data.get("relationship", "unknown"),
                RelationshipType.UNKNOWN
            )

            return ClassificationResult(
                relationship=relationship,
                shared_entity=data.get("shared_entity"),
                shared_event=data.get("shared_event"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "")
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return ClassificationResult(
                relationship=RelationshipType.UNKNOWN,
                shared_entity=None,
                shared_event=None,
                confidence=0.0,
                reasoning=f"Parse error: {e!s}"
            )

    async def group_markets(
        self,
        markets: list[PolymarketMarket]
    ) -> list[list[int]]:
        """
        Use LLM to group related markets by index.

        Args:
            markets: List of markets to group

        Returns:
            List of lists of market indices that belong together
        """
        if len(markets) < 2:
            return [[i] for i in range(len(markets))]

        # Build prompt
        questions = "\n".join(
            f"{i}. {m.question}"
            for i, m in enumerate(markets)
        )
        prompt = FAMILY_GROUPING_PROMPT.format(questions=questions)

        # Check cache
        if self.config.cache_enabled:
            cached = self._cache.get(prompt)
            if cached is not None:
                self._cache_hits += 1
                groups: list[list[int]] = cached
                return groups

        # Call LLM
        self._call_count += 1

        try:
            response = await self._provider.complete(prompt, self.config)
            groups = self._parse_grouping_response(response, len(markets))

            # Cache result
            if self.config.cache_enabled:
                self._cache.set(prompt, groups)

            return groups

        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
            # Return each market as its own group
            return [[i] for i in range(len(markets))]

    def _parse_grouping_response(
        self,
        response: str,
        num_markets: int
    ) -> list[list[int]]:
        """Parse LLM grouping response"""
        try:
            # Extract JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            groups = []
            seen = set()

            for group in data.get("groups", []):
                indices = [i for i in group.get("indices", []) if 0 <= i < num_markets]
                if indices:
                    groups.append(indices)
                    seen.update(indices)

            # Add ungrouped as singletons
            for i in range(num_markets):
                if i not in seen:
                    groups.append([i])

            return groups

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse grouping response: {e}")
            return [[i] for i in range(num_markets)]

    def get_stats(self) -> dict[str, Any]:
        """Get classifier statistics"""
        return {
            "call_count": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._call_count + self._cache_hits),
            "provider": self.config.provider,
            "model": self.config.model,
        }

    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()


# =============================================================================
# FACTORY
# =============================================================================

def create_classifier(
    provider: str = "anthropic",
    use_mock: bool = False
) -> LLMClassifier:
    """
    Create an LLM classifier instance.

    Args:
        provider: "anthropic" or "openai"
        use_mock: If True, use mock provider for testing

    Returns:
        Configured LLMClassifier
    """
    config = LLMConfig(provider="mock") if use_mock else LLMConfig(provider=provider)

    return LLMClassifier(config)
