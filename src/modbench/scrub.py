"""Persona-marker scrubbing for sensitivity analysis.

Question: are embedding-based diversity metrics inflated by persona-specific
surface markers (first-person pronouns, "as a teacher", "my lesson plans")
rather than reflecting true content diversity?

We compare metrics computed on:
  - original opinions (no scrub)
  - light_scrub: remove obvious persona markers (sentence-initial "I think",
    "as a X" prefixes, "My X" → "The X")
  - aggressive_scrub: also remove all first-person pronouns

If ranking across conditions is preserved, persona-marker leakage is bounded.
If ranking changes, the original metric primarily measured surface markers.
"""

import hashlib
import re

# === Light scrub: remove obvious persona markers ===

_LIGHT_PATTERNS = [
    # Sentence-initial epistemic phrases ("I think", "I believe", etc.)
    (re.compile(r"^\s*(I|i)\s+(think|believe|feel|see|find|worry|fear|hope|wonder|guess|suspect|argue|maintain|hold|reckon)\s+(that\s+)?", re.MULTILINE), ""),
    # "as a/an X[,]" prefix clauses
    (re.compile(r"^\s*[Aa]s\s+an?\s+[A-Za-z][A-Za-z\s\-]*[,]\s*", re.MULTILINE), ""),
    (re.compile(r",\s*as\s+an?\s+[A-Za-z][A-Za-z\s\-]*[,]\s*", re.IGNORECASE), ", "),
    # ORDER MATTERS: specific phrases first, then generic "my X" → "the X" replacements
    # "From my perspective" / "In my view" / "In my experience" — remove entirely
    (re.compile(r"\b(?:from|in)\s+my\s+(?:perspective|view|opinion|experience|mind|case)[,\s]+", re.IGNORECASE), ""),
    # Generic "My X" → "The X"
    (re.compile(r"\b[Mm]y\s+([a-z][a-z\-]*)", re.MULTILINE), r"the \1"),
    # "Our X" → "The X"
    (re.compile(r"\b[Oo]ur\s+([a-z][a-z\-]*)", re.MULTILINE), r"the \1"),
]


def light_scrub(text: str) -> str:
    """Light scrub: remove obvious persona markers but keep most content intact."""
    out = text
    for pattern, repl in _LIGHT_PATTERNS:
        out = pattern.sub(repl, out)
    # Collapse extra whitespace + capitalize first letter
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


# === Aggressive scrub: also remove all first-person pronouns ===

_AGGRESSIVE_PATTERNS = [
    # "When I/we ..." up to next clause boundary (comma or period)
    (re.compile(r"\bwhen\s+(?:I|we)\s+[^,\.;]+[,;]\s*", re.IGNORECASE), ""),
    # All first-person pronouns (kept simple — let embedding handle residual)
    (re.compile(r"\b(?:I|me|myself|we|us|ourselves)\b", re.IGNORECASE), ""),
    # Possessive forms
    (re.compile(r"\b(?:my|mine|our|ours)\b", re.IGNORECASE), ""),
]


def aggressive_scrub(text: str) -> str:
    """Aggressive scrub: light + remove all first-person pronouns."""
    out = light_scrub(text)
    for pattern, repl in _AGGRESSIVE_PATTERNS:
        out = pattern.sub(repl, out)
    # Clean up double spaces, leading punctuation, etc.
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"^\s*[,;\.]+\s*", "", out)
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


# === Provenance ===

def scrub_config_hash() -> dict:
    """Hash of the scrub patterns for reproducibility tracking."""
    light_str = "|".join(p.pattern for p, _ in _LIGHT_PATTERNS)
    agg_str = "|".join(p.pattern for p, _ in _AGGRESSIVE_PATTERNS)
    return {
        "light_hash": hashlib.sha256(light_str.encode()).hexdigest()[:8],
        "aggressive_hash": hashlib.sha256((light_str + agg_str).encode()).hexdigest()[:8],
        "light_n_patterns": len(_LIGHT_PATTERNS),
        "aggressive_n_patterns": len(_LIGHT_PATTERNS) + len(_AGGRESSIVE_PATTERNS),
    }
