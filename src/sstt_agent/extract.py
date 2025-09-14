import re
from typing import List, Tuple
from .canonical import (
    make_event_phrase,
    make_concept_phrase,
    make_thing_phrase,
    normalize_space,
)

Relation = Tuple[str, str, str, str, str]

CAUSE_PATTERNS = [
    r"(?P<src>[^.;:]+?)\s+(?:causes?|leads to|results in|drives|triggers|induces|enables?|creates?)\s+(?P<dst>[^.;:]+)",
    r"(?:which|that)\s+(?:causes?|leads to|results in|drives|triggers|induces|enables?|creates?|simplifies|improves|protects)\s+(?P<dst2>[^.;:]+)",
    r"(?P<src>[^.;:]+?)\s+(?:provides?|offers?|delivers?)\s+(?P<dst>[^.;:]+?)\s+(?:which|that)\s+(?P<dst2>[^.;:]+)",
]
EXPRESS_PATTERNS = [
    r"(?P<src>[^.;:]+?)\s+(?:express(?:es|ed)?|reflects|signals|indicates|represents|shows?|demonstrates?)\s+(?P<dst>[^.;:]+)",
]

ACTION_VERBS = {
    "reduce",
    "decrease",
    "lower",
    "curb",
    "cut",
    "increase",
    "raise",
    "grow",
    "expand",
    "widen",
    "narrow",
    "tighten",
    "shrink",
    "compress",
}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_phrase(p: str) -> str:
    p = re.sub(
        r"^\b(which|that|to|and|so|then|thus|therefore)\b[\s,]*",
        "",
        p.strip(),
        flags=re.I,
    )
    p = p.strip(",;: ")
    return p


def parse_action_triplets(sent: str) -> List[Relation]:
    rels: List[Relation] = []
    pattern = re.compile(
        rf"(?P<subj>[\w\s\-/]+?)\s+(?P<verb>{'|'.join(ACTION_VERBS)})\s+(?P<obj>[\w\s\-/]+)",
        re.I,
    )
    for m in pattern.finditer(sent):
        subj = clean_phrase(m.group("subj"))
        verb = m.group("verb").lower()
        obj = clean_phrase(m.group("obj"))
        if not subj or not obj:
            continue
        ev = make_event_phrase(verb, obj)
        subj_label = make_thing_phrase(subj)
        rels.append((subj_label, "LEADS-TO", normalize_space(ev), "thing", "event"))
    return rels


def parse_cause(sent: str) -> List[Relation]:
    rels: List[Relation] = []
    for pat in CAUSE_PATTERNS:
        for m in re.finditer(pat, sent, flags=re.I):
            src = m.groupdict().get("src", "")
            dst = m.groupdict().get("dst") or m.groupdict().get("dst2") or ""
            if not dst:
                continue
            if re.search(r"\bnarrow\w*\b", dst, flags=re.I):
                dst_event = make_event_phrase("narrow", dst)
            else:
                dst_event = make_event_phrase("effect", dst)
            if src:
                src_event = make_event_phrase("cause", src)
            else:
                src_event = make_event_phrase("cause", "Cause")
            rels.append(
                (
                    normalize_space(src_event),
                    "LEADS-TO",
                    normalize_space(dst_event),
                    "event",
                    "event",
                )
            )
    return rels


def parse_express(sent: str) -> List[Relation]:
    rels: List[Relation] = []
    for pat in EXPRESS_PATTERNS:
        for m in re.finditer(pat, sent, flags=re.I):
            src = clean_phrase(m.group("src"))
            dst = clean_phrase(m.group("dst"))
            if not src or not dst:
                continue
            if re.search(r"\bnarrow\w*\b", src, flags=re.I):
                src_event = make_event_phrase("narrow", src)
            else:
                src_event = src.title()
            concept_head = dst
            prep_obj = None
            m_in = re.search(r"(.+?)\s+in\s+(.+)", dst, flags=re.I)
            m_of = re.search(r"(.+?)\s+of\s+(.+)", dst, flags=re.I)
            if m_in:
                concept_head, prep_obj = m_in.group(1), m_in.group(2)
            elif m_of:
                concept_head, prep_obj = m_of.group(1), m_of.group(2)
            concept = make_concept_phrase(concept_head, prep_obj)
            rels.append(
                (
                    normalize_space(src_event),
                    "EXPRESSES",
                    normalize_space(concept),
                    "event",
                    "concept",
                )
            )
    return rels


def extract_relations(text: str) -> List[Relation]:
    rels: List[Relation] = []
    for sent in split_sentences(text):
        sent = sent.strip()
        if not sent:
            continue
        rels.extend(parse_action_triplets(sent))
        rels.extend(parse_cause(sent))
        rels.extend(parse_express(sent))
    seen = set()
    uniq: List[Relation] = []
    for r in rels:
        key = (r[0], r[1], r[2])
        if key not in seen:
            uniq.append(r)
            seen.add(key)
    return uniq
