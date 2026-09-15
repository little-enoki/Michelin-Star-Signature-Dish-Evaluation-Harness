#!/usr/bin/env python3
"""
Eval harness: "Michelin NYC signature dish" hallucination + faithfulness test.

WHAT THIS TESTS
----------------
Input:  a cuisine type (French, Italian, Contemporary, Korean, Japanese)
Output: a model's claim about the signature dish(es) of Michelin-recognized
        NYC restaurants matching that cuisine.

This mirrors exactly the trust problem Vincent (Clio's legal AI) is built to
solve: a confident, fluent answer is worthless — even dangerous — if it isn't
grounded in a verifiable source. Here the "verifiable source" is the Michelin
Guide instead of case law, but the eval mechanics are the same one you'd use
to test whether a legal AI product is fabricating citations.

METRICS (deliberately mirrors real RAG/legal-AI eval metrics)
---------------------------------------------------------------
1. Restaurant coverage  -> like "context recall": of the real, known
   restaurants for this cuisine, how many did the model surface at all?
2. Dish faithfulness     -> like "faithfulness"/"groundedness": for restaurants
   the model DID name, does the dish it describes actually match the real
   signature dish, or is it close-sounding fabrication?
3. Unverified mentions   -> like "citation hallucination": restaurants named
   by the model that don't appear in our reference set at all. NOTE: our
   reference set is a sample, not the full Michelin Guide, so this is a
   "flag for manual review," not an automatic hallucination verdict --
   exactly how a real eval pipeline should treat low-confidence flags.

USAGE
-----
1. Ask a model (ChatGPT, Claude, Gemini, Vincent, whatever you're testing)
   this exact question for each cuisine:

       "What are the signature dish(es) of Michelin-recognized restaurants
        in New York City that serve {cuisine} cuisine? Name the restaurant
        and the specific dish."

2. Paste its raw answer into responses/<cuisine>.txt (lowercase filename,
   e.g. responses/french.txt). One file per cuisine.

3. Run:  python3 eval_harness.py

4. Read the printed report and eval_report.json for the full breakdown.

The repo ships with SYNTHETIC example responses already in responses/ so you
can see the pipeline run end-to-end before you plug in a real model. Replace
those files with real pasted model output to run a genuine test.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
GROUND_TRUTH_PATH = BASE_DIR / "ground_truth.json"
RESPONSES_DIR = BASE_DIR / "responses"
REPORT_PATH = BASE_DIR / "eval_report.json"

DISH_MATCH_THRESHOLD = 0.20  # word-overlap ratio above which we call a dish "grounded"
GENERIC_NAME_WORDS = {"sushi", "le", "the", "el", "la", "new", "york", "restaurant"}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard overlap of normalized word tokens -- cheap, dependency-free
    stand-in for a semantic similarity / faithfulness score."""
    ta, tb = set(normalize(a).split()), set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def best_dish_match(claimed_text: str, true_dishes: list[str]):
    """Word-overlap (Jaccard) only -- deliberately NOT blended with
    difflib's character-level ratio. Character-level similarity gives
    partial credit to same-length, similarly-worded sentences even when
    the actual content (the dish) is completely different -- which lets
    fabricated answers slip past as 'grounded'. That failure mode is real
    and worth knowing: it's the reason production eval systems use
    LLM-as-judge scoring for faithfulness instead of plain string
    similarity. See README for a worked example."""
    best_score, best_dish = 0.0, None
    for dish in true_dishes:
        score = token_overlap_ratio(claimed_text, dish)
        if score > best_score:
            best_score, best_dish = score, dish
    return best_score, best_dish


def split_into_segments(response_text: str) -> list[str]:
    """Break the response into line/sentence-level chunks so we can score a
    restaurant's dish claim against ONLY the text about that restaurant,
    instead of diluting the match against the entire response."""
    lines = [l.strip(" -\t") for l in response_text.splitlines() if l.strip()]
    segments = []
    for line in lines:
        segments.extend(re.split(r"(?<=[.!?])\s+", line))
    return [s for s in segments if s]


def distinctive_key(norm_name: str) -> str:
    """Pick the most distinctive word in a restaurant name to use for partial
    matching (e.g. 'Yasuda' for 'Sushi Yasuda', not the generic 'Sushi' --
    which would also match 'Sushi Nakazawa' and 'Sushi Sho'). Falls back to
    the full name if every word is generic or the name is a single word."""
    words = [w for w in norm_name.split(" ") if w not in GENERIC_NAME_WORDS and len(w) > 2]
    if not words:
        return norm_name
    return max(words, key=len)


def find_restaurant_mentions(response_text: str, restaurant_names: list[str]):
    """Which known restaurant names appear (fuzzily) in the response text,
    and the specific segment(s) of text that mention them. Matches on the
    full name first; only falls back to a single distinctive word so that
    restaurants sharing a generic prefix (e.g. 'Sushi Yasuda' / 'Sushi Sho')
    don't get conflated with each other."""
    segments = split_into_segments(response_text)
    mentioned = {}
    for name in restaurant_names:
        norm_name = normalize(name)
        key = distinctive_key(norm_name)
        # word-boundary matches only -- a plain substring check would let
        # "Octo" match inside "octopus", or "Noksu" match inside a longer
        # made-up word. \b works cleanly here because normalize() already
        # collapsed everything to lowercase alphanumeric + single spaces.
        name_pattern = re.compile(r"\b" + re.escape(norm_name) + r"\b")
        key_pattern = re.compile(r"\b" + re.escape(key) + r"\b")
        matching_segments = [
            seg for seg in segments
            if name_pattern.search(normalize(seg)) or key_pattern.search(normalize(seg))
        ]
        if matching_segments:
            mentioned[name] = " ".join(matching_segments)
    return mentioned


LEADING_NAME_PATTERN = re.compile(
    r"^([A-Z][A-Za-z'\-]+(?:\s[A-Z][A-Za-z'\-]+){0,4})"
    r"(?:'s\s|\s+(?:is|are|serves|offers|specializes)\b)"
)
GENERIC_TERMS = {"michelin", "new york", "michelin guide", "michelin guide new york", "nyc"}


def extract_unverified_restaurant_candidates(response_text: str, known_names: list[str]):
    """Heuristic entity extraction: look for 'Capitalized Phrase is/serves/'s...'
    patterns at the start of each line -- the shape a model uses when it's
    naming a restaurant. Flags names that don't match our reference set.
    This is intentionally simple (a production harness would use NER or an
    LLM-as-judge instead) and is a REVIEW flag, not an automatic hallucination
    verdict -- our reference set is a sample of the Michelin Guide, not the
    full list, so a flagged name may just be a real restaurant we didn't
    include, and needs a human to check."""
    known_norm = {normalize(n) for n in known_names}
    unverified = []
    for segment in split_into_segments(response_text):
        m = LEADING_NAME_PATTERN.match(segment)
        if not m:
            continue
        candidate = m.group(1).strip()
        nc = normalize(candidate)
        if not nc or nc in GENERIC_TERMS:
            continue
        if any(nc in kn or kn in nc for kn in known_norm):
            continue
        unverified.append(candidate)
    # de-dupe, preserve order
    seen = set()
    out = []
    for c in unverified:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def grade_cuisine(cuisine: str, response_text: str, ground_truth_entries: list[dict]) -> dict:
    known_names = [e["restaurant"] for e in ground_truth_entries]
    mentioned = find_restaurant_mentions(response_text, known_names)  # name -> matching segment text

    per_restaurant = []
    for name, segment_text in mentioned.items():
        entry = next(e for e in ground_truth_entries if e["restaurant"] == name)
        score, matched_dish = best_dish_match(segment_text, entry["dishes"])
        per_restaurant.append({
            "restaurant": name,
            "grounded": score >= DISH_MATCH_THRESHOLD,
            "match_score": round(score, 2),
            "closest_true_dish": matched_dish,
            "source": entry["source"],
        })

    coverage = len(mentioned) / len(known_names) if known_names else 0.0
    grounded_count = sum(1 for r in per_restaurant if r["grounded"])
    faithfulness = grounded_count / len(per_restaurant) if per_restaurant else None

    unverified = extract_unverified_restaurant_candidates(response_text, known_names)

    return {
        "cuisine": cuisine,
        "known_restaurants_in_reference_set": len(known_names),
        "restaurants_mentioned": len(mentioned),
        "coverage_of_reference_set": round(coverage, 2),
        "dish_faithfulness_rate": round(faithfulness, 2) if faithfulness is not None else None,
        "per_restaurant_detail": per_restaurant,
        "unverified_mentions_flagged_for_review": unverified,
    }


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    reports = []

    for cuisine, entries in ground_truth.items():
        response_file = RESPONSES_DIR / f"{cuisine.lower()}.txt"
        if not response_file.exists():
            print(f"[skip] no response file for {cuisine} (expected {response_file.name})")
            continue
        response_text = response_file.read_text()
        report = grade_cuisine(cuisine, response_text, entries)
        reports.append(report)

    REPORT_PATH.write_text(json.dumps(reports, indent=2))

    print("\n=== MICHELIN NYC SIGNATURE DISH EVAL ===\n")
    header = f"{'Cuisine':<14}{'Coverage':<11}{'Faithfulness':<14}{'Flagged (review)':<18}"
    print(header)
    print("-" * len(header))
    for r in reports:
        faith = "n/a" if r["dish_faithfulness_rate"] is None else f"{r['dish_faithfulness_rate']*100:.0f}%"
        print(f"{r['cuisine']:<14}{r['coverage_of_reference_set']*100:>4.0f}%     {faith:<14}{len(r['unverified_mentions_flagged_for_review']):<18}")

    print(f"\nFull detail written to {REPORT_PATH.relative_to(BASE_DIR)}\n")

    for r in reports:
        if r["unverified_mentions_flagged_for_review"]:
            print(f"[{r['cuisine']}] flagged for manual review (not auto-failed): "
                  f"{r['unverified_mentions_flagged_for_review']}")
        for detail in r["per_restaurant_detail"]:
            if not detail["grounded"]:
                print(f"[{r['cuisine']}] LOW FAITHFULNESS: model's claim about '{detail['restaurant']}' "
                      f"only scored {detail['match_score']} against closest real dish "
                      f"(\"{detail['closest_true_dish']}\") -- check source: {detail['source']}")


if __name__ == "__main__":
    main()
