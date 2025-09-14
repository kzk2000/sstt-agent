import re


def titleize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()


def singularize_simple(word: str) -> str:
    if word.lower().endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.lower().endswith("sses"):
        return word[:-2]
    if word.lower().endswith("s") and not word.lower().endswith("ss"):
        return word[:-1]
    return word


def gerund(lemma: str) -> str:
    lemma = lemma.strip()
    if not lemma:
        return lemma
    if lemma.endswith("e") and lemma not in ("be", "see"):
        return lemma[:-1] + "ing"
    if lemma.endswith("ie"):
        return lemma[:-2] + "ying"
    return lemma + "ing"


def make_event_phrase(verb: str, obj_text: str | None = None) -> str:
    v = verb.lower().strip() if verb else ""
    if obj_text:
        obj = titleize(obj_text)
        if v in {"reduce", "decrease", "lower", "curb", "cut"}:
            return f"{obj} Reduction"
        if v in {"increase", "raise", "grow", "expand", "widen"}:
            return f"{obj} Increase"
        if v in {"narrow", "tighten", "shrink", "compress"}:
            parts = obj.split()
            if parts:
                last = parts[-1]
                if last.lower().endswith("ies") and len(last) > 3:
                    parts[-1] = last[:-3] + "y"
                elif last.lower().endswith("sses"):
                    parts[-1] = last[:-2]
                elif last.lower().endswith("s") and not last.lower().endswith("ss"):
                    parts[-1] = last[:-1]
            obj_sing = " ".join(parts)
            return f"{obj_sing} Narrowing"
        return f"{titleize(gerund(v))} {obj}"
    return titleize(gerund(v))


def make_concept_phrase(head_text: str, prep_obj_text: str | None = None) -> str:
    head = titleize(head_text) if head_text else ""
    if prep_obj_text:
        return f"{titleize(prep_obj_text)} {head}"
    return head


def make_thing_phrase(text: str) -> str:
    return titleize(text)


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()
