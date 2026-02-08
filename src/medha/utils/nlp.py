"""NLP utilities: parameter extraction and keyword overlap scoring."""

import re
import logging
from typing import Dict, List
from collections import defaultdict

from medha.types import QueryTemplate
from medha.exceptions import ParameterExtractionError

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by",
})


class ParameterExtractor:
    """Extract parameters from user questions using a cascading strategy."""

    def __init__(self, use_spacy: bool = True):
        """Initialize the extractor.

        Args:
            use_spacy: If True, attempt to load a spaCy model for NER.
                Falls back gracefully if spaCy is not installed.
        """
        self._nlp = None
        self._spacy_available = False

        if use_spacy:
            self._try_load_spacy()

    @property
    def spacy_available(self) -> bool:
        """Whether spaCy NER is available."""
        return self._spacy_available

    def extract(self, question: str, template: QueryTemplate) -> Dict[str, str]:
        """Extract all parameters for a template from a question.

        Cascading strategy:
            1. Regex patterns from template.parameter_patterns
            2. spaCy NER (if available)
            3. Fallback heuristics (numbers, capitalized words)

        Args:
            question: The raw user question.
            template: The matched QueryTemplate.

        Returns:
            Dict mapping parameter names to extracted values.
            Only fully extracted results are returned (all params present).

        Raises:
            ParameterExtractionError: If not all required parameters
                can be extracted.
        """
        if not template.parameters:
            return {}

        # 1. Regex patterns from template
        params = self._extract_via_regex(question, template)
        if len(params) == len(template.parameters):
            return params

        # 2. spaCy NER
        if self._spacy_available:
            spacy_params = self._extract_via_spacy(question, template)
            for key, value in spacy_params.items():
                if key not in params:
                    params[key] = value
            if len(params) == len(template.parameters):
                return params

        # 3. Heuristic fallback
        heuristic_params = self._extract_via_heuristics(question, template)
        for key, value in heuristic_params.items():
            if key not in params:
                params[key] = value

        if len(params) == len(template.parameters):
            return params

        missing = set(template.parameters) - set(params)
        raise ParameterExtractionError(
            f"Could not extract parameters {missing} from: {question!r}"
        )

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy + regex fallback.

        Returns a dict mapping entity type to list of values:
            - "number": ["5", "100"]
            - "person": ["John Smith"]
            - "org": ["Acme Corp"]
            - "cardinal": ["5"]

        Always includes regex-extracted numbers and capitalized words,
        even when spaCy is available (union of both).
        """
        entities: Dict[str, List[str]] = defaultdict(list)

        if self._spacy_available and self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "CARDINAL"):
                    entities[ent.label_.lower()].append(ent.text)

        # Regex fallback (always applied — union with spaCy)
        numbers = re.findall(r"\b\d+\b", text)
        names = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b", text)

        if numbers:
            entities["number"].extend(numbers)
        if names:
            entities["person"].extend(names)

        return dict(entities)

    def render_query(
        self, template: QueryTemplate, parameters: Dict[str, str]
    ) -> str:
        """Inject parameters into a query template.

        Applies basic sanitization: only allows alphanumeric, spaces,
        hyphens, and underscores in parameter values.

        Args:
            template: The QueryTemplate with placeholders.
            parameters: Extracted parameter values.

        Returns:
            The rendered query string.

        Raises:
            ParameterExtractionError: If a parameter value fails sanitization
                or a placeholder remains unfilled.
        """
        query = template.query_template

        for param, value in parameters.items():
            safe_value = self._sanitize_value(value)
            if not safe_value:
                raise ParameterExtractionError(
                    f"Parameter {param!r} value {value!r} is empty after sanitization"
                )
            query = query.replace(f"{{{param}}}", safe_value)

        # Check for remaining unfilled placeholders
        remaining = re.findall(r"\{(\w+)\}", query)
        if remaining:
            raise ParameterExtractionError(
                f"Unfilled placeholders in rendered query: {remaining}"
            )

        return query

    # --- Private methods ---

    def _try_load_spacy(self) -> None:
        """Attempt to load a spaCy model. Try multiple models in order."""
        models = ["en_core_web_sm", "en_core_web_md", "en"]
        for model_name in models:
            try:
                import spacy

                self._nlp = spacy.load(model_name)
                self._spacy_available = True
                logger.info("Loaded spaCy model: %s", model_name)
                return
            except (OSError, ImportError):
                continue
        logger.info("spaCy not available; using regex-only extraction")

    def _extract_via_regex(
        self, question: str, template: QueryTemplate
    ) -> Dict[str, str]:
        """Extract parameters using template-defined regex patterns."""
        params: Dict[str, str] = {}
        if not template.parameter_patterns:
            return params

        for param_name, pattern in template.parameter_patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                match = matches[0]
                params[param_name] = match if isinstance(match, str) else match[0]

        return params

    def _extract_via_spacy(
        self, question: str, template: QueryTemplate
    ) -> Dict[str, str]:
        """Extract parameters using spaCy NER."""
        entities = self.extract_entities(question)
        params: Dict[str, str] = {}

        for param in template.parameters:
            if param in ("count", "number") and "number" in entities:
                params[param] = entities["number"][0]
            elif param in ("user", "person", "name") and "person" in entities:
                params[param] = entities["person"][0]
            elif param in ("company", "org", "organization") and "org" in entities:
                params[param] = entities["org"][0]
            elif param == "project":
                project_match = re.search(
                    r"project\s+([A-Za-z0-9_]+)", question, re.IGNORECASE
                )
                if project_match:
                    params[param] = project_match.group(1)

        return params

    def _extract_via_heuristics(
        self, question: str, template: QueryTemplate
    ) -> Dict[str, str]:
        """Fallback: extract numbers and capitalized words."""
        params: Dict[str, str] = {}
        numbers = re.findall(r"\b\d+\b", question)
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", question)

        numeric_params = ("count", "number", "limit", "top")
        for param in template.parameters:
            if param in params:
                continue
            if param in numeric_params and numbers:
                params[param] = numbers.pop(0)
            elif param not in numeric_params and capitalized:
                params[param] = capitalized.pop(0)

        return params

    @staticmethod
    def _sanitize_value(value: str) -> str:
        """Remove characters that could cause injection.

        Allows: alphanumeric, spaces, hyphens, underscores.
        """
        return re.sub(r"[^\w\s-]", "", str(value))


def keyword_overlap_score(question: str, template_text: str) -> float:
    """Calculate keyword overlap between a question and a template.

    Removes stop words and template placeholders before comparison.

    Args:
        question: Normalized user question.
        template_text: Template pattern text.

    Returns:
        Float in [0.0, 1.0] representing the fraction of template
        keywords found in the question.
    """
    question_words = set(re.findall(r"\b\w+\b", question.lower()))
    template_words = set(re.findall(r"\b\w+\b", template_text.lower()))

    question_words -= _STOP_WORDS
    template_words -= _STOP_WORDS
    template_words = {w for w in template_words if not (w.startswith("{") and w.endswith("}"))}

    if not template_words:
        return 0.0

    intersection = question_words & template_words
    return len(intersection) / len(template_words)
