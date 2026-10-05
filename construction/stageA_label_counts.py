"""
Source-specific handlers for the eight datasets, plus a label counting tool.

Each source dataset has its own structure (different field names, different ways
of marking what a sample is about), so each one has a handler function here that
knows how to read it. A handler takes a sample and returns the target label, its
kind (disease, pest, weed, symptom description, object or stress condition, or a
non-groundable kind such as action, cause, healthy or plant identification) and
the host crop when it can be determined. The handlers are imported by
final_preprocess.py and capture_nongroundable.py, which is where the routing into
the groundable and non-groundable streams happens. Adding a dataset means adding
one handler function.

Run on its own, the script counts the unique crops, diseases, pests, weeds,
symptoms and stress conditions per dataset and prints a report. It changes no data.

A note on precision: the counts for crops, diseases, pests, weeds and stress
conditions were checked against the data several times. The finer split between
action, symptom_description and cause (mainly in AgroBench's management text)
is less exact: management advice comes in too many sentence patterns to catch
every one with fixed phrase lists, and roughly one entry in five of those three
buckets lands in a neighboring bucket. This does not affect whether a sample is
recognized as being about a disease, pest or weed, only which of those three
non-groundable buckets it is sorted into.

Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
"""

import json
import os
import re
from collections import defaultdict

JSONL_DIR = "./work/unified_jsonl"
OUTPUT_PATH = "./work/stage_a_label_counts.json"

# How many samples to look at per dataset. AgroBench and AgroCoT are small
# enough that we look at every single sample (None means "no limit"). CDDM
# is over a million samples, so we only look at a large slice of it.
SAMPLE_LIMITS = {
    "agrobench": None,
    "agrocot":   None,
    "cddm":      None,  # read everything now, not just 100k
    "agmmu":     None,
    "agromind":  None,
    "leafbench": None,
    "leafnet":   None,
    "mirage":    None,
}


# ---------------------------------------------------------------------------
# SHARED BUILDING BLOCKS
# These are small pieces every dataset handler can use, so we don't repeat
# the same logic three times.
# ---------------------------------------------------------------------------

# A list of crop names we know to look for when a dataset doesn't tell us
# the crop directly and we have to find it by reading the text.
KNOWN_CROPS = [
    "apple", "tomato", "potato", "grape", "corn", "wheat", "rice", "soybean",
    "strawberry", "peach", "pepper", "cucumber", "lettuce", "cherry", "plum",
    "blueberry", "raspberry", "orange", "grapefruit", "pumpkin", "eggplant",
    "broccoli", "radish", "sorghum", "barley", "cotton", "sugarcane", "bean",
    "alfalfa", "banana", "avocado", "almond", "apricot", "artichoke",
    "asparagus", "basil", "beet", "blackberry", "olive", "mango", "papaya",
    "pecan", "pineapple", "nectarine", "walnut", "fig", "lemon", "lime",
    "spinach", "cabbage", "carrot", "onion", "garlic", "squash", "melon",
    "watermelon", "kiwi", "maize", "millet", "oat", "rye", "sunflower",
    "tea", "coffee", "cocoa", "rubber", "tobacco", "hemp",
]

# Words that suggest an answer is naming a living pest (an insect, mite,
# etc.) rather than a disease or weed.
PEST_KEYWORDS = [
    "aphid", "beetle", "mite", "moth", "caterpillar", "weevil",
    "whitefly", "thrips", "borer", "scale", "fly", "worm",
    "hopper", "bug", "mealybug", "springtail", "cutworm",
]

# Words that suggest an answer is naming a weed.
WEED_KEYWORDS = [
    "weed", "invasive", "dandelion", "thistle", "nutsedge",
    "bindweed", "ragweed", "horsenettle", "pigweed",
]

# Words that suggest the answer is saying the plant is healthy, i.e. there
# is no disease/pest/weed present at all.
HEALTHY_KEYWORDS = ["healthy", "no disease", "normal", "no issue"]

# Words that suggest the answer is an environmental/nutritional stress
# condition rather than a living disease, pest, or weed.
STRESS_KEYWORDS = [
    "deficiency", "drought", "heat stress", "frost", "sunscald", "sunburn",
    "waterlog", "nutrient", "chlorosis", "cold damage", "salt stress",
    "overwater", "underwater", "edema", "catface", "zippering",
]

# Action words. If an answer contains one of these, it is probably telling
# the reader to DO something (prune, spray, remove) rather than naming
# what is wrong with the plant. This matters because actions are not
# something you can draw a box around in an image.
# NOTE: this list was expanded after checking real output and finding
# action sentences (e.g. "bag the bunches using perforated polyethylene",
# "burying infected plants to prevent insect feeding") that were slipping
# past the original shorter list and landing in symptom_description by
# mistake.
ACTION_KEYWORDS = [
    "pull", "remove", "spray", "apply", "prune", "cut",
    "dig", "discard", "treat with", "use a", "avoid",
    "handpick", "pick off", "destroy", "rotate",
    "collect fallen", "manage the", "bag", "bury", "burying",
    "store", "plant only", "plant resistant", "select resistant",
    "select healthy", "ensure", "improve", "increase", "reduce",
    "limit", "maintain", "fumigate", "disinfect", "sterilize",
    "wrap", "cover", "drain", "irrigate", "fertiliz",
    "fungicide", "myclobutanil", "copper-based", "chemical treatment",
    "use ", "using ", "apply ", "treat ", "inoculate ",
    "use ", "using ", "apply ", "treat ", "inoculate ",
]


def find_crop_in_text(text):
    """Look through a piece of text for any crop name we recognize."""
    text_lower = text.lower()
    for crop in KNOWN_CROPS:
        if crop in text_lower:
            return crop
    return None


def looks_like_measurement(text):
    """
    Check if a piece of text is mostly numbers, percent signs, and units
    rather than an actual word-based label. Things like '80% - 90%' or
    '[5000-5100]square meters' should say yes here.
    """
    stripped = text.lower().strip()
    if not stripped:
        return False
    has_digit = any(ch.isdigit() for ch in stripped)
    if not has_digit:
        return False
    allowed_chars = "%.-–~ °cmkginftsquare"
    matching_chars = sum(1 for ch in stripped if ch.isdigit() or ch in allowed_chars)
    ratio = matching_chars / len(stripped)
    return ratio > 0.6


def categorize_generic_text(text):
    """
    A general-purpose first guess at what KIND of thing a piece of text
    is describing: healthy, pest, weed, stress_condition,
    measurement_or_instruction, action, symptom_description, or disease
    (the fallback when nothing else matches).
    This is used as a shared helper inside each dataset's own handler, but
    each handler also adds its own extra checks on top of this, since each
    dataset has its own quirks.
    """
    text_lower = text.lower()

    if any(keyword in text_lower for keyword in HEALTHY_KEYWORDS):
        return "healthy"
    if looks_like_measurement(text):
        return "measurement_or_instruction"
    if any(keyword in text_lower for keyword in ACTION_KEYWORDS):
        return "action"
    if any(keyword in text_lower for keyword in PEST_KEYWORDS):
        return "pest"
    if any(keyword in text_lower for keyword in WEED_KEYWORDS):
        return "weed"
    if any(keyword in text_lower for keyword in STRESS_KEYWORDS):
        return "stress_condition"
    if len(text_lower.split()) > 5:
        return "symptom_description"
    return "disease"


# ---------------------------------------------------------------------------
# AGROBENCH HANDLER
# AgroBench tells us directly, in its metadata, which "subset" each sample
# belongs to. We checked every single one of these subsets by hand, so we
# know exactly what each one contains:
#   did = disease identification (the answer IS a clean disease name)
#   pid = pest identification (the answer IS a clean pest name)
#   wid = weed identification (the answer IS a clean weed name)
#   dmn = disease management advice (symptoms + actions, mixed together)
#   cmn = crop management advice (measurements + some actions, mixed)
#   tm  = traditional farming tools/methods (nothing to do with disease/pest/weed)
#   mqa = farm machinery questions (nothing to do with disease/pest/weed)
# ---------------------------------------------------------------------------

def handle_agrobench_sample(sample):
    """
    Look at one AgroBench sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely (not relevant to crops/diseases/pests/weeds).
    """
    answer = sample.get("answer", "")
    question = sample.get("question", "")
    if not answer or not isinstance(answer, str):
        return None

    metadata = sample.get("metadata", {}) or {}
    subset = metadata.get("source_subset", "")
    crop = metadata.get("crop")
    category = metadata.get("category", "")

    # The two "irrelevant" subsets: traditional tools and farm machinery.
    # Neither has anything to do with crops, diseases, pests, or weeds.
    if subset in ("tm", "mqa"):
        return None

    # Disease identification: the answer is already a clean disease name,
    # and the category field tells us if it's caused by a fungus, bacteria,
    # virus, or an environmental/nutritional problem.
    if subset == "did":
        category_lower = category.lower()
        if category_lower in ("environmental", "nutritional"):
            kind = "stress_condition"
        else:
            kind = "disease"
        return {"kind": kind, "label": answer.strip(), "crop": crop}

    # Pest identification: the answer is already a clean pest name.
    if subset == "pid":
        return {"kind": "pest", "label": answer.strip(), "crop": crop}

    # Weed identification: the answer is already a clean weed name.
    if subset == "wid":
        return {"kind": "weed", "label": answer.strip(), "crop": crop}

    # Disease management advice: this turned out to have more variety than
    # just symptoms and actions. Checking real output found body-part
    # questions ("what part of the plant"), cause/environment questions
    # ("what environmental condition favors this"), and action verbs that
    # a keyword list could never fully cover ("lime the soil"). Reading the
    # actual QUESTION text turned out to be far more reliable than guessing
    # from the answer's wording.
    if subset == "dmn":
        question_lower = question.lower()
        answer_lower = answer.lower()

        # "What part of the plant..." -> answer names a plant part, not a
        # symptom or action. This is object-kind content.
        if "what part of" in question_lower:
            return {"kind": "object", "label": answer.strip(), "crop": crop}

        # Questions about WHY something happens, or what ENVIRONMENTAL/
        # WEATHER condition causes or favors it, are asking about the
        # underlying cause, not the visible symptom or the recommended
        # action. This list was expanded after a second round of checking
        # real output and finding more weather/spread-condition phrasings
        # than the first pass caught (e.g. "what kind of weather...",
        # "which condition is most likely to promote the spread...",
        # "tend to become more severe").
        cause_question_markers = [
            "why is it", "why does", "why should", "environmental condition",
            "what factor", "factor contributes", "under what condition",
            "under what kind", "what condition is most likely", "what causes",
            "what kind of weather", "what kind of condition", "kind of environment",
            "promote the spread", "spread more easily", "spread more readily",
            "tend to spread", "tend to become", "become more severe",
            "more likely to spread", "more likely to occur", "more likely to develop",
            "more likely to appear", "favors the spread", "favors the development",
            "favors the emergence", "helps this disease spread",
            "weather makes this disease", "weather helps", "what increases the risk",
            "what increases the likelihood", "could make problems", "more likely",
            "how does the pathogen", "how does the disease spread",
            "how can problems like", "how does it spread",
            "how is it transmitted", "what condition increases", "what condition makes", 
            "what weather", "under which condition", "what conditions is most likely",
            "what condition is most favorable", "what condition encourages",
            "which conditions is most likely to prevent",
        ]
        if any(marker in question_lower for marker in cause_question_markers):
            return {"kind": "cause", "label": answer.strip(), "crop": crop}

        # Questions explicitly about management, practice, or what should
        # be done are asking for an action, regardless of which verb the
        # answer happens to use. Expanded the same way as the cause list
        # above, after finding more real phrasings on a second pass
        # ("which timing or practice", "when should...be applied", "what
        # practice can help prevent", "what is a good action").
        action_question_markers = [
            "management practice", "practice should", "practices is",
            "farming practice", "what should be done", "how can the",
            "how can this", "what is the recommended", "what method",
            "how should", "what strategy", "what is a good way",
            "what is an effective", "what is one way", "how does plowing",
            "how does pruning", "how does removing", "how does irrigation",
            "what soil amendment", "helps lower", "helps reduce",
            "what action", "what is recommended", "timing or practice",
            "when should", "practice can help", "what is a good action",
            "what is a good practice", "what is good practice",
            "what is a recommended", "what is not recommended",
            "what should not", "what action helps", "what action is",
            "cultural practice", "cultural practices", "planting site",
            "planting location", "kind of practice", "type of practice",
            "how can proper", "how can we", "what is not a good way",
            "when is the best time", "what practice helps", "how long can",
            "how long does", "how do you manage", "how do you prevent",
            "how do you control", "what is a common way",
            "what fungicide", "what chemical", "which fungicide", 
            "what treatment", "what fungicides is", "where should receive",
            "when managing", "what potential impact", "how frequently should", "how often should",
        ]
        if any(marker in question_lower for marker in action_question_markers):
            return {"kind": "action", "label": answer.strip(), "crop": crop}

        # As a backup, still check the answer text itself for action verbs,
        # since some real actions show up under more generic-sounding
        # questions than the markers above.
        if any(keyword in answer_lower for keyword in ACTION_KEYWORDS):
            return {"kind": "action", "label": answer.strip(), "crop": crop}

        # Anything left is treated as a symptom/diagnosis description,
        # which matches what's actually left over once the categories
        # above are pulled out (e.g. "what symptom is characteristic of...").
        return {"kind": "symptom_description", "label": answer.strip(), "crop": crop}

    # Crop management advice: a mix of pure measurements and real actions.
    if subset == "cmn":
        if looks_like_measurement(answer):
            return None  # not useful for our taxonomy, skip it
        answer_lower = answer.lower()
        if any(keyword in answer_lower for keyword in ACTION_KEYWORDS):
            return {"kind": "action", "label": answer.strip(), "crop": crop}
        return None

    # Anything else with an unrecognized subset: skip rather than guess.
    return None


# ---------------------------------------------------------------------------
# AGROCOT HANDLER
# AgroCoT does not have a simple "subset" field like AgroBench. Instead it
# has three numbers per sample: type_id, dimension_id, sub_dimension_id.
# We found, by reading thousands of real examples, that type_id tells us
# the SHAPE of the question (counting? yes/no? open-ended? multi-image
# picking?) regardless of topic, while dimension_id/sub_dimension_id tells
# us the TOPIC. We need both: type_id to throw away unreliable formats,
# and dimension_id/sub_dimension_id to know which kind of label we're
# looking at among the ones that are left.
#
# type_id values that are NEVER useful for our taxonomy (counting,
# yes/no answers, multi-image file-path answers, coordinates, severity
# ratings, etc). NOTE: type_id 7 (severity/occlusion ratings like
# "Moderate occlusion") was identified during verification but was
# missing from this set in an earlier version of this script - confirmed
# by checking real output that it was leaking through. Fixed here.
AGROCOT_SKIP_TYPE_IDS = {2, 3, 4, 5, 7, 8, 9, 11}
# ---------------------------------------------------------------------------

def handle_agrocot_sample(sample):
    """
    Look at one AgroCoT sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    answer = sample.get("answer", "")
    question = sample.get("question", "")
    if not answer or not isinstance(answer, str):
        return None

    metadata = sample.get("metadata", {}) or {}
    type_id = metadata.get("type_id")
    dimension_id = metadata.get("dimension_id")
    sub_dimension_id = metadata.get("sub_dimension_id")

    # First filter: throw away question formats we already know are
    # unreliable for taxonomy purposes (counts, yes/no, file-paths, etc).
    if type_id in AGROCOT_SKIP_TYPE_IDS:
        return None

    crop = find_crop_in_text(answer)
    answer_lower = answer.lower()

    # dimension_id 1 is always counting, regardless of type_id. Skip it.
    if dimension_id == 1:
        return None

    # dimension_id 2: spatial/area topics. Only sub_dimension_id 3 has real
    # action-style content (the rest is coordinates and measurements).
    if dimension_id == 2:
        if sub_dimension_id == 3 and not looks_like_measurement(answer):
            if any(keyword in answer_lower for keyword in ACTION_KEYWORDS):
                return {"kind": "action", "label": answer.strip(), "crop": crop}
        return None

    # dimension_id 3: species/appearance topics.
    if dimension_id == 3:
        if sub_dimension_id == 1:
            # Plant species identification -- this is "what is it",
            # not "what's wrong with it", so it's an object, not a disease.
            return {"kind": "object", "label": answer.strip(), "crop": crop}
        if sub_dimension_id == 3:
            # This bucket is more mixed than it first looked: real
            # symptom/diagnosis content sits alongside damage-cause
            # questions, growth-advice questions, and generic short
            # answers that are NOT disease names ("general", "glyphosate
            # damage"). Found by checking real output, not assumed.
            # Because of this, we do NOT trust the generic "default to
            # disease" fallback here -- we only accept an answer as a
            # disease/symptom if it actually looks like one.
            if any(keyword in answer_lower for keyword in ACTION_KEYWORDS):
                return {"kind": "action", "label": answer.strip(), "crop": crop}
            if any(keyword in answer_lower for keyword in HEALTHY_KEYWORDS):
                return {"kind": "healthy", "label": "healthy", "crop": crop}
            if any(keyword in answer_lower for keyword in PEST_KEYWORDS):
                return {"kind": "pest", "label": answer.strip(), "crop": crop}
            if any(keyword in answer_lower for keyword in WEED_KEYWORDS):
                return {"kind": "weed", "label": answer.strip(), "crop": crop}
            if any(keyword in answer_lower for keyword in STRESS_KEYWORDS):
                return {"kind": "stress_condition", "label": answer.strip(), "crop": crop}
            # Reject vague single-word non-answers that aren't real labels
            # (this check was present in the dimension_id 4 version of this
            # logic but was missing here -- found by checking real output
            # and seeing a bare "Yes" slip through).
            if answer_lower.strip() in ("general", "none", "n/a", "unknown", "unclear",
                                          "yes", "no"):
                return None
            # Only accept as a symptom/diagnosis if the question itself is
            # actually asking about symptoms, condition, or disease -- not
            # asking about growth tips, vegetation type, or anything else.
            question_lower = question.lower()
            symptom_question_markers = ["symptom", "condition", "disease", "damage", "affect"]
            if any(marker in question_lower for marker in symptom_question_markers):
                return {"kind": "symptom_description", "label": answer.strip(), "crop": crop}
            return None  # doesn't clearly look like disease/symptom content, skip rather than guess
        return None  # sub_dimension_id 2 is a mixed bucket, skip for now

    # dimension_id 4: this is AgroCoT's main disease/pest/symptom area.
    if dimension_id == 4:
        if sub_dimension_id == 1:
            return None  # confirmed unreliable mix, always skip
        if sub_dimension_id == 2:
            # Most answers are pest names.
            # Exception: skip management/soil answers that slipped through
            if answer_lower.startswith('it helps') or answer_lower.startswith('handpick'):
                return {"kind": "action", "label": answer.strip(), "crop": crop}
            return {"kind": "pest", "label": answer.strip(), "crop": crop}
        if sub_dimension_id == 3:

            # Extract disease name from "No, it has X" answers
            # e.g. "No, it has Black Rot Fungus" -> "black rot"
            import re as _re
            _no_it_has = _re.match(
                r'^no,?\s+it\s+has\s+(.+?)(?:\s+fungus|\s+virus|\s+disease|\s+water\s+mold|\s+bacteria)?$',
                answer_lower.strip()
            )
            if _no_it_has:
                extracted = _no_it_has.group(1).strip()
                # Fix known abbreviations and dataset errors
                _no_it_has_fixes = {
                    'ylcv': 'yellow leaf curl virus',
                    'tomv': 'tomato mosaic virus',
                    'polysora': 'southern corn rust',
                    'scarab': 'scab',
                    'greening june': 'citrus greening',
                    'target spot bacteria': 'target spot',
                    'late blight water mold': 'late blight',
                }
                extracted = _no_it_has_fixes.get(extracted, extracted)
                return {"kind": "symptom_description", "label": extracted, "crop": crop}


            # Checking real output found the same problem here as in
            # dimension 3/subdim 3: vague non-answers like "general" and
            # status questions about unrelated things ("Grass,which is
            # crowded with grass" answering a question about a TREE's
            # status) were being defaulted to "disease" by the generic
            # fallback. Applying the same stricter check used above.
            if any(keyword in answer_lower for keyword in ACTION_KEYWORDS):
                return {"kind": "action", "label": answer.strip(), "crop": crop}
            if any(keyword in answer_lower for keyword in HEALTHY_KEYWORDS):
                return {"kind": "healthy", "label": "healthy", "crop": crop}
            if any(keyword in answer_lower for keyword in PEST_KEYWORDS):
                return {"kind": "pest", "label": answer.strip(), "crop": crop}
            if any(keyword in answer_lower for keyword in WEED_KEYWORDS):
                return {"kind": "weed", "label": answer.strip(), "crop": crop}
            if any(keyword in answer_lower for keyword in STRESS_KEYWORDS):
                return {"kind": "stress_condition", "label": answer.strip(), "crop": crop}
            # Reject vague single-word non-answers that aren't real labels.
            if answer_lower.strip() in ("general", "none", "n/a", "unknown", "unclear",
                                          "yes", "no"):
                return None
            question_lower = question.lower()
            symptom_question_markers = ["symptom", "condition", "disease", "damage", "affect", "pest"]
            if any(marker in question_lower for marker in symptom_question_markers):
                return {"kind": "symptom_description", "label": answer.strip(), "crop": crop}
            return None  # doesn't clearly look like disease/symptom content, skip rather than guess
        return None

    # dimension_id 5: management/tools/actions.
    if dimension_id == 5:
        if sub_dimension_id == 1:
            return None  # traditional tools, not relevant

        if sub_dimension_id in (2, 3):
            # Checking real output revealed this bucket is NOT purely
            # actions -- it also contains traditional-farming-method
            # trivia ("Agroforestry", "Ancient Rome") and ecosystem/
            # vegetation-type questions ("Central Andean puna"), neither
            # of which is a disease/pest/weed management action.
            question_lower = question.lower()

            # Ecosystem/vegetation-type questions: skip these outright,
            # they're geography trivia, not farming actions.
            if "vegetation type" in question_lower or "ecosystem" in question_lower:
                return None

            # Traditional-method identification questions ask WHAT something
            # is/is called, not HOW to do something -- that's identification,
            # not an action, even though the topic is farming methods.
            identification_markers = ["what is the name", "what is the method",
                                       "which ancient", "which traditional",
                                       "what method is shown", "famously depicted"]
            if any(marker in question_lower for marker in identification_markers):
                return None

            # What's left genuinely looks like a real action/practice
            # recommendation (e.g. "use a scuffle hoe", "pull the
            # seedlings out").
            return {"kind": "action", "label": answer.strip(), "crop": crop}
        return None

    # Any other dimension_id we haven't seen before: skip rather than guess.
    return None


# ---------------------------------------------------------------------------
# CDDM HANDLER
# CDDM is different from the other two datasets: it has almost no useful
# metadata (just "subset" and "split"), so everything has to be figured out
# by reading the answer text itself. We verified that CDDM's answers
# follow a small number of fixed patterns:
#   "...afflicted with X" / "...affected by X" / "...infected with X"  -> a disease name
#   "...healthy"                                                       -> no disease present
#   "This is a/an X leaf"  (with no disease mentioned)                 -> just naming the plant, an object
# We also verified that pest names and weed names essentially never show
# up cleanly in CDDM on their own (the one real pest we found, Spider
# Mites, gets caught by the same disease-style pattern below).
# ---------------------------------------------------------------------------

CDDM_DISEASE_PATTERN = re.compile(
    r"(?:afflicted with|affected by|infected with|exhibits symptoms of|has)\s+"
    r"([A-Z][A-Za-z\s]+?)(?:[\.,]|$)"
)

CDDM_PLANT_ID_PATTERN = re.compile(
    r"^(?:this is|no, this is|this is not)\b", re.IGNORECASE
)


def handle_cddm_sample(sample):
    """
    Look at one CDDM sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    answer = sample.get("answer", "")
    if not answer or not isinstance(answer, str):
        return None

    crop = find_crop_in_text(answer)
    answer_lower = answer.lower()

    if "healthy" in answer_lower:
        return {"kind": "healthy", "label": "healthy", "crop": crop}

    disease_match = CDDM_DISEASE_PATTERN.search(answer)
    if disease_match:
        disease_name = disease_match.group(1).strip()
        return {"kind": "disease", "label": disease_name, "crop": crop}

    # No disease pattern matched. If the answer is just naming the plant
    # ("This is an apple leaf"), that's an object, not a disease.
    if CDDM_PLANT_ID_PATTERN.match(answer.strip()):
        return {"kind": "object", "label": answer.strip(), "crop": crop}

    # Anything else we don't recognize: skip rather than guess.
    return None



# ---------------------------------------------------------------------------
# LEAFBENCH HANDLER
# LeafBench is one of the cleanest datasets we have. Every sample has a
# clear question_type field in its metadata that tells us exactly what kind
# of content it is. We verified this by reading all 13,950 samples.
#
# question_type values and what they mean:
#   DC  = Disease Classification: "Which specific disease can you identify?"
#         Answer format: "Crop Disease" e.g. "Apple Black rot"
#         We split crop from disease using a verified list of crop names.
#         NOTE: "Black Pepper" is a two-word crop -- this is the only
#         multi-word crop in all 86 unique DC answers, verified by reading
#         every single unique answer in the full dataset.
#   CSI = Crop Species Identification: "Which plant species is visible?"
#         Answer is a clean crop name e.g. "Apple", "Tomato"
#   SI  = Symptom Identification: "What is the primary symptom observed?"
#         Answer is a real visual symptom description
#   HDC = Health/Disease Classification: "Does image show a diseased plant?"
#         Answer is just "Yes" or "No" -- not useful as a label, skip
#   PC  = Pathogen Classification: "Which type of pathogen?"
#         Answer is "Fungal", "Bacterial" etc -- category metadata, skip
#   SNC = Scientific Name Classification: "What is the scientific name?"
#         Answer is a scientific name, not a common disease name, skip
# ---------------------------------------------------------------------------

# All crops that appear as the first word(s) in LeafBench DC answers.
# Verified by reading every single unique DC answer (86 total).
# "Black Pepper" is the only two-word crop -- all others are one word.
LEAFBENCH_TWO_WORD_CROPS = ["black pepper"]


def split_leafbench_dc_answer(answer):
    """
    Split a LeafBench DC answer like "Apple Black rot" into
    (crop, disease) pair. Handles the one two-word crop
    "Black Pepper" as a special case.
    Returns (crop_str, disease_str) both lowercase.
    """
    answer_lower = answer.lower().strip()

    # Check two-word crops first
    for two_word in LEAFBENCH_TWO_WORD_CROPS:
        if answer_lower.startswith(two_word + " "):
            disease = answer[len(two_word):].strip()
            return two_word, disease.lower()

    # All other crops are one word -- split on first space
    parts = answer.split(" ", 1)
    if len(parts) == 2:
        return parts[0].lower(), parts[1].lower()
    return answer.lower(), ""


def handle_leafbench_sample(sample):
    """
    Look at one LeafBench sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    answer = sample.get("answer", "")
    if not answer or not isinstance(answer, str):
        return None

    metadata = sample.get("metadata", {}) or {}
    qt = metadata.get("question_type", "")
    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()

    # DC: Disease Classification
    # Answer format is "Crop Disease" e.g. "Apple Black rot"
    # Some answers are "Crop Healthy" -- those are healthy cases
    if qt == "DC":
        crop, disease = split_leafbench_dc_answer(answer_clean)
        if not disease or disease == "healthy":
            return {"kind": "healthy", "label": "healthy", "crop": crop}
        return {"kind": "disease", "label": disease, "crop": crop}

    # CSI: Crop Species Identification -- answer is a clean crop name
    if qt == "CSI":
        return {
            "kind": "object",
            "label": answer_clean,
            "crop": answer_clean.lower(),
        }

    # SI: Symptom Identification -- answer is a real visual symptom
    if qt == "SI":
        return {
            "kind": "symptom_description",
            "label": answer_clean,
            "crop": None,
        }

    # HDC, PC, SNC -- not useful for taxonomy label extraction, skip
    return None







# ---------------------------------------------------------------------------
# AGMMU HANDLER
# AGMMU is a real farmer Q&A dataset with 772 samples. Every sample has a
# clear qtype field in its metadata. We verified all 772 samples by reading
# the full dump.
#
# qtype values and what they mean:
#   disease/issue identification: clean disease name as answer
#   insect/pest:                  clean pest name as answer
#   symptom/visual description:   describes what the damage looks like
#   management instructions:      tells you what to do, not groundable
#   species:                      MIXED -- could be plant species (object),
#                                 pest/insect, animal, fungus, weed etc.
#
# The species qtype has 126 unique answers spanning insects, plants, weeds,
# animals, fungi, snakes, birds and rodents. We use an explicit whitelist
# of confirmed pest/insect answers (verified by reading all 126 unique
# answers) rather than keyword matching, because keyword matching
# incorrectly classified plants like "lantana", "beeblossom", and
# "giant sumpweed" as pests.
#
# Anything in the species qtype that is NOT in the pest whitelist is
# classified as object (plant/animal/fungus species identification).
# "None" answers are skipped.
# ---------------------------------------------------------------------------

# Confirmed pest/insect species answers from AGMMU species qtype.
# Built by manually reviewing all 126 unique species answers.
AGMMU_PEST_SPECIES_WHITELIST = {
    "spider", "caterpillar", "grub/caterpillar", "bee", "wasp",
    "wheel bug", "camel cricket", "beetle", "butterfly", "earwig",
    "bark beetle", "wolf spider", "monarch butterfly", "imperial moth",
    "giant ichneumon wasp", "caddisfly", "elm seed bug",
    "american carrion beetle", "hercules beetle", "carolina locust",
    "question mark butterfly", "giant water bug", "cross spider",
    "praying mantis", "jumping spider", "leaffooted bug",
    "house centipede", "cicada", "assassin bug", "european hornet",
    "slug",
}


def handle_agmmu_sample(sample):
    """
    Look at one AGMMU sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    answer = sample.get("answer", "")
    if not answer or not isinstance(answer, str):
        return None

    metadata = sample.get("metadata", {}) or {}
    qtype = metadata.get("qtype", "")
    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()

    # Skip empty or None answers
    if answer_lower in ("none", ""):
        return None

    # Disease/issue identification -- answer is a clean disease name
    if qtype == "disease/issue identification":
        if any(kw in answer_lower for kw in STRESS_KEYWORDS):
            return {"kind": "stress_condition",
                    "label": answer_clean, "crop": None}
        return {"kind": "disease", "label": answer_clean, "crop": None}

    # Pest identification -- answer is a clean pest name
    if qtype == "insect/pest":
        return {"kind": "pest", "label": answer_clean, "crop": None}

    # Symptom/visual description -- describes visible damage
    if qtype == "symptom/visual description":
        return {"kind": "symptom_description",
                "label": answer_clean, "crop": None}

    # Management instructions -- cannot be grounded in an image, skip
    if qtype == "management instructions":
        return None

    # Species -- use whitelist to distinguish pests from plants/animals.
    # Anything not in the whitelist is classified as object since it is
    # a species identification (plant, animal, fungus etc).
    if qtype == "species":
        if answer_lower in AGMMU_PEST_SPECIES_WHITELIST:
            return {"kind": "pest", "label": answer_clean, "crop": None}
        return {"kind": "object", "label": answer_clean, "crop": None}

    # Any unrecognised qtype: skip rather than guess
    return None







# ---------------------------------------------------------------------------
# LEAFNET HANDLER
# LeafNet has a single question template used for every sample:
# "Describe the disease symptoms visible in this leaf image."
#
# All answers follow one fixed format:
# "a image of {Crop} {healthy/leaves} diseased by {Disease}
#  with symptoms of {symptom description}"
# OR for healthy cases:
# "a image of {Crop} healthy leaves with leaves appearing normal"
#
# We verified this by reading 10,000 samples (representative slice of
# 121,337 total). There are exactly 10 unique answers in the dataset --
# a classification dataset disguised as VQA.
#
# One dataset quirk: "Sugarcane leaves diseased by Healthy Leaves" uses
# the word "diseased by" followed by "Healthy Leaves" -- this is a real
# labeling error in the original dataset. We treat it as healthy.
#
# The answers already contain the crop name, disease name, AND symptom
# description in one pre-formatted string. We extract all three.
# ---------------------------------------------------------------------------

# Pattern to extract crop, disease, and symptom from LeafNet answers
LEAFNET_PATTERN = re.compile(
    r"a image of ([A-Za-z ]+?) (?:healthy leaves|leaves diseased by "
    r"([A-Za-z ]+?) with symptoms of (.+))$",
    re.IGNORECASE
)


def handle_leafnet_sample(sample):
    """
    Look at one LeafNet sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    answer = sample.get("answer", "")
    if not answer or not isinstance(answer, str):
        return None

    answer_lower = answer.lower().strip()

    # Healthy cases -- two patterns:
    # "a image of X healthy leaves with leaves appearing normal"
    # "a image of X leaves diseased by Healthy Leaves with symptoms of..."
    if "healthy" in answer_lower:
        crop_match = re.search(
            r"a image of ([A-Za-z ]+?) (?:healthy|leaves)", answer, re.IGNORECASE
        )
        crop = crop_match.group(1).strip().lower() if crop_match else None
        return {"kind": "healthy", "label": "healthy", "crop": crop}

    # Disease cases -- extract crop, disease, and symptom
    match = LEAFNET_PATTERN.match(answer.strip())
    if match:
        crop    = match.group(1).strip().lower()
        disease = match.group(2).strip().lower() if match.group(2) else None
        symptom = match.group(3).strip() if match.group(3) else None

        if not disease:
            return None

        # Return the disease as the primary label.
        # The symptom description is stored separately -- it is the
        # richest visual description in the entire dataset and will
        # be used as the grounding_prompt when we expand the taxonomy.
        return {
            "kind":    "disease",
            "label":   disease,
            "crop":    crop,
            "symptom": symptom,  # extra field -- richer than any other dataset
        }

    return None








# ---------------------------------------------------------------------------
# AGROMIND HANDLER
# AgroMind has 28,482 samples across 13 task codes. Most tasks are NOT
# grounding-relevant (counting, measurements, boundary coordinates, spatial
# relationships, climate zones). We verified this by reading all 28,482
# samples.
#
# Task codes and our decision for each:
#   CO  = Counting (how many trees/plots?) -> SKIP, answer is a number
#   AS  = Area/Size measurements -> SKIP, answer is a measurement
#   BD  = Boundary detection (bounding box coordinates) -> SKIP
#   SC  = Spatial counting (which image has most?) -> SKIP, letter answers
#   SR  = Spatial relationship (above/below/left/right) -> SKIP
#   CTR = Climate/terrain recognition -> SKIP, not agricultural
#   VPR = Visual property (crown diameter, tassel counts) -> SKIP
#   PL  = Precision farming decisions -> action kind
#   OC  = Object/Crop identification -> crop name or pest name
#         BUT: skip Yes/No answers (617+613 samples have no useful label)
#         AND: skip scientific names like "Vitis", "Puccinia"
#   PDD = Plant Disease Detection -> disease name
#         BUT: skip severity answers ("serious", "general", "negligible")
#         AND: extract disease from "No, it has X" pattern (same as AgroCoT)
#   SL  = Spatial anomaly localization -> anomaly type (weed_cluster etc)
#         BUT: skip spatial location answers (Top-Left, Central region etc)
#   AR  = Anomaly Recognition -> anomaly type for single-image non-letter
#         BUT: skip Yes/No answers
#   GSR = General Scene Recognition -> healthy status only
#         BUT: skip number answers (tree counts) and measurements
#
# Multi-image letter answers (is_multiple_image=True AND answer in A-E)
# are ALWAYS skipped regardless of task -- these are broken comparison
# questions where the answer is just a letter pointing to an image.
# ---------------------------------------------------------------------------

# Anomaly type labels from AgroMind aerial/satellite imagery.
# These are the only valid non-skip answers for SL and AR tasks.
AGROMIND_ANOMALY_TYPES = {
    "weed_cluster", "waterway", "double_plant",
    "cloud_shadow", "standing_water", "planter_skip",
}

# Pattern for extracting disease names from AgroMind PDD answers
# that follow the "No, it has X" format (same as AgroCoT)
AGROMIND_NO_IT_HAS_PATTERN = re.compile(
    r"^no,?\s+it\s+has\s+(.+?)(?:\s+fungus|\s+virus|\s+water\s+mold|\s+mold|\s+bacteria|\s+water)?$",
    re.IGNORECASE
)

# Severity words that appear in PDD answers but are not disease names
AGROMIND_SEVERITY_WORDS = {
    "serious", "general", "healthy", "yes", "no",
    "almost negligible", "negligible",
}

# Words that indicate a scientific name rather than a common label
AGROMIND_SCIENTIFIC_INDICATORS = [
    "puccinia", "vitis", "helminthosporium", "cercospora",
    " sp.", " spp.", " pv.",
]


def handle_agromind_sample(sample):
    """
    Look at one AgroMind sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    meta = sample.get("metadata", {}) or {}
    task = meta.get("task", "")
    is_multi = meta.get("is_multiple_image", False)

    # Answer can be integer in some AgroMind samples -- always convert
    raw_answer = sample.get("answer", "") or ""
    answer_clean = str(raw_answer).strip()
    answer_lower = answer_clean.lower()

    # Skip multi-image letter answers -- these are broken comparison
    # questions where the answer is just a letter (A/B/C/D/E)
    if is_multi and answer_clean in ("A", "B", "C", "D", "E"):
        return None

    # Skip entirely irrelevant task codes
    if task in ("CO", "AS", "BD", "SC", "SR", "CTR", "VPR"):
        return None

    # PL: Precision farming decisions
    # Most PL answers are quantitative descriptions like
    # "a large amount,extremely limited,extremely high" or
    # foreign language crop names -- these are not useful
    # action labels. Only keep answers that look like real
    # management recommendations (contain action keywords).
    if task == "PL":
        if answer_lower in ("", "yes", "no"):
            return None
        # Skip quantitative comma-separated descriptions
        if "," in answer_clean and any(
            w in answer_lower for w in
            ["amount", "large", "limited", "reasonable",
             "high", "low", "normal", "extremely"]
        ):
            return None
        # Skip French language entries -- AgroMind has a French-language
        # subset with crop/farming labels in French that are not useful
        # action labels for our taxonomy.
        french_indicators = [
            "autre ", "légume", "fourrage", "graminée",
            "légumineuse", "trèfle", "banane -", "agrume",
            "verger", "pérenne", "annuel", "chanvre",
            "café", "cacao", "concombre", "cornichon",
            "aubergine", "poivron", "piment", "oignon",
            "échalote", "ornementale", "maïs", "pomme de terre",
            "geranium", "soft winter", "forage crops",
            "oil seeds", "horticulture",
        ]
        if any(fi in answer_lower for fi in french_indicators):
            return None
        # Skip single foreign/scientific words with no spaces
        if len(answer_clean.split()) <= 2 and not any(
            kw in answer_lower for kw in ACTION_KEYWORDS
        ):
            return None
        return {"kind": "action", "label": answer_clean, "crop": None}

    # OC: Object/Crop identification
    if task == "OC":
        # Skip Yes/No answers -- no useful label
        if answer_lower in ("yes", "no"):
            return None
        # Skip scientific names
        if any(ind in answer_lower for ind in AGROMIND_SCIENTIFIC_INDICATORS):
            return None
        # Check if it's a pest name
        crop = find_crop_in_text(answer_clean)
        if any(kw in answer_lower for kw in PEST_KEYWORDS):
            return {"kind": "pest", "label": answer_clean, "crop": None}
        # Otherwise it's a crop/object identification
        return {"kind": "object", "label": answer_clean, "crop": crop}

    # PDD: Plant Disease Detection
    if task == "PDD":
        # Skip severity answers and bare Yes/No
        if answer_lower in AGROMIND_SEVERITY_WORDS:
            return None
        if answer_lower.startswith("it's diseased"):
            return None
        # Healthy
        if "healthy" in answer_lower:
            return {"kind": "healthy", "label": "healthy", "crop": None}
        # "No, it has X" pattern -- extract disease name
        match = AGROMIND_NO_IT_HAS_PATTERN.match(answer_lower)
        if match:
            disease = match.group(1).strip()
            # Apply same fixes we used in AgroCoT A2 cleanup
            disease_fixes = {
                "late blight water mold": "late blight",
                "late blight water":      "late blight",
                "target spot bacteria":   "target spot",
                "ylcv":                   "yellow leaf curl virus",
                "greening june":          "citrus greening",
                "polysora":               "southern corn rust",
                "scarab":                 "scab",
                "dwarf mosaic":           "dwarf mosaic virus",
                "spider mite damage":     "spider mite damage",
                "black measles":          "black measles",
                "spotting leaf spot":     "spotting leaf spot",
                "tomv":                   "tomato mosaic virus",
            }
            disease = disease_fixes.get(disease, disease)
            return {"kind": "disease", "label": disease, "crop": None}
        # Any other PDD answer we don't recognise: skip
        return None

    # SL: Spatial anomaly localization
    if task == "SL":
        # Only keep known anomaly type labels
        # Skip spatial location answers (Top-Left, Central region etc)
        # and "No anomaly" / "No distribution"
        if answer_lower in ("no anomaly", "no distribution", "no"):
            return {"kind": "healthy", "label": "no anomaly", "crop": None}
        if answer_clean in AGROMIND_ANOMALY_TYPES:
            if answer_clean == "weed_cluster":
                return {"kind": "weed", "label": answer_clean, "crop": None}
            return {"kind": "object", "label": answer_clean, "crop": None}
        return None  # spatial location answer, skip

    # AR: Anomaly Recognition
    if task == "AR":
        # Skip Yes/No
        if answer_lower in ("yes", "no"):
            return None
        # Handle compound answers like "waterway, weed_cluster"
        # Split and check each part
        parts = [p.strip() for p in answer_clean.split(",")]
        valid_parts = [p for p in parts if p in AGROMIND_ANOMALY_TYPES]
        if not valid_parts:
            return None
        # Use first valid anomaly type as the primary label
        label = valid_parts[0]
        if label == "weed_cluster":
            return {"kind": "weed", "label": label, "crop": None}
        return {"kind": "object", "label": label, "crop": None}

    # GSR: General Scene Recognition
    if task == "GSR":
        # Only keep "Healthy" -- skip numbers (tree counts) and measurements
        if answer_lower == "healthy":
            return {"kind": "healthy", "label": "healthy", "crop": None}
        return None

    # Any unrecognised task code: skip
    return None












# ---------------------------------------------------------------------------
# MIRAGE HANDLER
# MIRAGE is a real farmer consultation dataset with 40,889 samples.
# Real farmers submitted questions with photos to agricultural extension
# services, and experts answered. We verified all 40,889 samples.
#
# Key facts verified from full dataset read:
# - Answers are long paragraphs (average 736 chars) -- we NEVER use the
#   answer text as a label. We use entity_name from metadata instead.
# - entity_name is empty for 3,973 samples -- skip those entirely.
# - 4 configs exist: MMST_Standard, MMMT_Direct, MMMT_Decomp,
#   MMST_Contextual. We skip MMMT_Decomp entirely -- it is a duplicate
#   of MMMT_Direct with decomposed sub-questions, same images and entities.
# - 8 categories. We only keep Identification categories (3 of 8):
#   Plant Disease Identification -> diagnosis/disease
#   Insect and Pest Identification -> diagnosis/pest
#   Plant Identification -> object
#   All Management/Guidance/Care categories -> skip (action kind,
#   not groundable, no entity to box in the image)
# ---------------------------------------------------------------------------

MIRAGE_KEEP_CATEGORIES = {
    "Plant Disease Identification":   "disease",
    "Insect and Pest Identification": "pest",
    "Plant Identification":           "object",
}


def handle_mirage_sample(sample):
    """
    Look at one MIRAGE sample and decide what kind of thing it is.
    Returns a dictionary describing the sample, or None if this sample
    should be skipped entirely.
    """
    meta = sample.get("metadata", {}) or {}

    # Skip MMMT_Decomp -- these are duplicates of MMMT_Direct
    if meta.get("config", "") == "MMMT_Decomp":
        return None

    category = meta.get("category", "")

    # Skip Management, Guidance, Care, Others categories
    if category not in MIRAGE_KEEP_CATEGORIES:
        return None

    # Use entity_name as the label -- NOT the long paragraph answer
    entity_name = meta.get("entity_name", "").strip()
    if not entity_name:
        return None  # skip samples with no clean entity name

    entity_name_lower = entity_name.lower()
    kind_category = MIRAGE_KEEP_CATEGORIES[category]

    # Try to find crop context from question text
    question = sample.get("question", "")
    crop = find_crop_in_text(question)

    if kind_category == "disease":
        return {
            "kind":  "disease",
            "label": entity_name_lower,
            "crop":  crop,
        }

    if kind_category == "pest":
        return {
            "kind":  "pest",
            "label": entity_name_lower,
            "crop":  crop,
        }

    if kind_category == "object":
        return {
            "kind":  "object",
            "label": entity_name_lower,
            "crop":  crop,
        }

    return None




















# ---------------------------------------------------------------------------
# PUTTING IT ALL TOGETHER
# This is the part that doesn't care which dataset it's looking at. It just
# calls whichever handler function matches the dataset name, collects the
# results, and counts them up. Adding a new dataset later means writing
# one new "handle_xxx_sample" function above and adding one line to this
# dictionary -- nothing below this point needs to change.
# ---------------------------------------------------------------------------


DATASET_HANDLERS = {
    "agrobench": handle_agrobench_sample,
    "agrocot":   handle_agrocot_sample,
    "cddm":      handle_cddm_sample,
    "leafbench": handle_leafbench_sample,
    "agmmu":     handle_agmmu_sample,
    "leafnet":   handle_leafnet_sample,
    "agromind":  handle_agromind_sample,
    "mirage":    handle_mirage_sample,
}


def characterize_dataset(dataset_name):
    """
    Read one dataset's JSONL file, run every sample through its matching
    handler function, and count up how many unique labels we find for
    each kind (disease, pest, weed, etc).
    """
    handler = DATASET_HANDLERS.get(dataset_name)
    if handler is None:
        print(f"  No handler defined for '{dataset_name}', skipping.")
        return None

    path = os.path.join(JSONL_DIR, f"{dataset_name}.jsonl")
    if not os.path.exists(path):
        print(f"  File not found: {path}, skipping.")
        return None

    limit = SAMPLE_LIMITS.get(dataset_name)

    # unique_labels_by_kind keeps a SET of unique label strings per kind,
    # so "Late Blight" mentioned 500 times still only counts once.
    unique_labels_by_kind = defaultdict(set)
    crops_seen = set()
    total_checked = 0
    total_skipped = 0

    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = handler(sample)
            total_checked += 1

            if result is None:
                total_skipped += 1
                continue

            kind = result["kind"]
            label = result["label"].strip().lower()[:100]  # cap length for sane reporting
            unique_labels_by_kind[kind].add(label)

            if result.get("crop"):
                crops_seen.add(result["crop"].strip().lower())

    # Build a clean summary, with a small sample of real labels for each
    # kind so we can eyeball whether the results look sensible.
    summary = {
        "dataset": dataset_name,
        "samples_checked": total_checked,
        "samples_skipped": total_skipped,
        "unique_crops": len(crops_seen),
        "crop_list": sorted(crops_seen)[:30],
    }
    for kind, labels in unique_labels_by_kind.items():
        summary[f"unique_{kind}_count"] = len(labels)
        summary[f"{kind}_sample"] = sorted(labels)[:20]

    return summary


def main():
    print(f"\n{'=' * 70}\nTAXONOMY CHARACTERIZATION (v2)\n{'=' * 70}\n")

    report = {}
    for dataset_name in DATASET_HANDLERS.keys():
        print(f"[Characterizing] {dataset_name} ...")
        result = characterize_dataset(dataset_name)
        if result:
            report[dataset_name] = result

    # Print a short summary table to the screen.
    print(f"\n{'=' * 70}")
    for dataset_name, summary in report.items():
        print(f"\n--- {dataset_name} ---")
        print(f"  samples checked: {summary['samples_checked']}  "
              f"(skipped: {summary['samples_skipped']})")
        print(f"  unique crops: {summary['unique_crops']}")
        for key, value in summary.items():
            if key.startswith("unique_") and key.endswith("_count"):
                kind_name = key.replace("unique_", "").replace("_count", "")
                print(f"  unique {kind_name}: {value}")
    print(f"\n{'=' * 70}\n")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full detailed report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()