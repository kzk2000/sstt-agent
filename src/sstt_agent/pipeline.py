from .extract import extract_relations
from .intent import compute_token_intent, phrase_intent
from .graph import N4LGraph


def doc_to_n4l(text: str) -> N4LGraph:
    token_scores = compute_token_intent(text)
    rels = extract_relations(text)
    g = N4LGraph()
    for src, etype, dst, src_type, dst_type in rels:
        si = phrase_intent(src, token_scores)
        di = phrase_intent(dst, token_scores)
        g.add_node(src, src_type, si)
        g.add_node(dst, dst_type, di)
        g.add_edge(src, etype, dst)
    return g
