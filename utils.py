import re
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


def legalize_noise_aware(tags: List[str],
                         valid_entity_types: set = None,
                         dangling_policy: str = "promote") -> List[str]:
    """Enforce IOB2 legality, choosing how to repair a dangling ``I-X`` (an
    ``I-`` tag with no valid same-type ``B-``/``I-`` predecessor).

    ``dangling_policy``:
      - ``"promote"`` (default): upgrade the dangling ``I-X`` to ``B-X``. Correct
        for BT/ATF, where a dropped ``B-`` should be recovered as a real entity.
      - ``"demote"``: drop the dangling ``I-X`` to ``O``. Correct for IF, where a
        residual dangling ``I-`` is a failed-merge artifact — promoting it would
        invent a false-positive single-token entity and collapse precision.

    Both policies produce SER=0 output; they differ only on dangling ``I-`` tags.
    Non-dangling tags, prefix normalization, and invalid-type stripping are
    identical across policies.
    """
    VALID_ENTITIES = valid_entity_types if valid_entity_types is not None \
                     else {"PER", "LOC", "ORG"}
    # Build a case-insensitive lookup: lowercase -> canonical form
    _valid_lower = {t.lower(): t for t in VALID_ENTITIES}
    fixed = []

    for i, tag in enumerate(tags):

        clean_tag = tag
        if clean_tag != "O":
            try:
                prefix, ent_type = clean_tag.split("-", 1)
                # Normalize prefix to uppercase
                prefix = prefix.upper()
                if prefix not in {"B", "I"}:
                    clean_tag = "O"
                else:
                    # Case-insensitive entity type lookup
                    ent_lower = ent_type.lower()
                    if ent_lower in _valid_lower:
                        clean_tag = f"{prefix}-{_valid_lower[ent_lower]}"
                    else:
                        clean_tag = "O"
            except ValueError:
                clean_tag = "O"


        if clean_tag.startswith("I-"):
            # Detect a dangling I- (no valid same-type predecessor).
            prev_tag = fixed[i-1] if i > 0 else "O"
            dangling = (
                prev_tag == "O"
                or prev_tag.split("-", 1)[1] != clean_tag.split("-", 1)[1]
            )
            if dangling:
                # demote -> drop the FP fragment; promote -> recover as B-.
                fixed.append("O" if dangling_policy == "demote"
                             else "B-" + clean_tag[2:])
                continue

        fixed.append(clean_tag)

    return fixed


def enforce_iob2_syntax(tags: List[str],
                       valid_entity_types: set = None) -> List[str]:
    """Promote-by-default IOB2 legalizer (unchanged behavior for all callers).

    Thin wrapper over :func:`legalize_noise_aware` with the historical
    ``dangling_policy="promote"`` semantics.
    """
    return legalize_noise_aware(tags, valid_entity_types,
                                dangling_policy="promote")

def extract_json_list(llm_output: str, fallback_length: int) -> list:
    if isinstance(llm_output, list):
        return llm_output
    try:
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_list = json.loads(json_str)
            if isinstance(parsed_list, list):
                return parsed_list
        parsed_list = json.loads(llm_output)
        if isinstance(parsed_list, list):
            return parsed_list
    except Exception as e:
        logger.warning(f"JSON parsing failed: {llm_output[:50]}...")
    return ["O"] * fallback_length