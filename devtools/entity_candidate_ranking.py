"""Non-authoritative candidate generation and ranking research primitives."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

_TOKEN = re.compile(r"\w+", re.UNICODE)


class RankingEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    graph_group: str = Field(min_length=1)


class RankingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    entity: RankingEntity
    relevance: int = Field(ge=0, le=1)
    category: str
    cross_document: bool = False


class RankingQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str
    query_entity: RankingEntity
    candidates: list[RankingCandidate]
    source_kind: str
    no_match: bool = False


class RankedCandidate(BaseModel):
    """Ranking metadata only: deliberately no identity/merge decision field."""
    model_config = ConfigDict(extra="forbid")
    query_entity_id: str
    candidate_entity_id: str
    rank: int = Field(gt=0)
    relevance_score: float
    candidate_sources: list[str]


def normalize_name(value: str) -> str:
    """Generic representation normalization; authoritative names are untouched."""
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_TOKEN.findall(folded))


def acronym(value: str) -> str:
    tokens = _TOKEN.findall(unicodedata.normalize("NFKC", value))
    return "".join(token[0] for token in tokens if token).casefold()


def lexical_score(left: str, right: str) -> float:
    a, b = normalize_name(left), normalize_name(right)
    ta, tb = set(a.split()), set(b.split())
    overlap = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    containment = min(len(a), len(b)) / max(len(a), len(b)) if a and (a in b or b in a) else 0.0
    return max(overlap, containment, SequenceMatcher(None, a, b).ratio())


def acronym_match(left: str, right: str) -> bool:
    a, b = normalize_name(left).replace(" ", ""), normalize_name(right).replace(" ", "")
    short_left=[t.casefold() for t in _TOKEN.findall(left) if len(t)>=2 and t.isupper()]
    short_right=[t.casefold() for t in _TOKEN.findall(right) if len(t)>=2 and t.isupper()]
    return ((len(a) >= 2 and a == acronym(right)) or (len(b) >= 2 and b == acronym(left)) or
            any(token == acronym(right) for token in short_left) or any(token == acronym(left) for token in short_right))


def candidate_sources(query: RankingEntity, candidate: RankingEntity, bge_rank: int, *, bge_cap: int = 10,
                      lexical_floor: float = .55) -> list[str]:
    if query.graph_group != candidate.graph_group:
        return []
    sources=[]
    if bge_rank <= bge_cap: sources.append("BGE")
    names_q=[query.canonical_name,*query.aliases]; names_c=[candidate.canonical_name,*candidate.aliases]
    if max(lexical_score(a,b) for a in names_q for b in names_c) >= lexical_floor: sources.append("LEXICAL")
    if any(acronym_match(a,b) for a in names_q for b in names_c): sources.append("ACRONYM")
    return sources


def rank_metrics(rankings: Sequence[Sequence[tuple[str, int]]], ks: Sequence[int]=(1,3,5,10,20)) -> dict:
    eligible=[items for items in rankings if any(rel for _,rel in items)]
    if not eligible:
        return {**{f"recall_at_{k}":0.0 for k in ks},"mrr":0.0,"ndcg_at_5":0.0,"ndcg_at_10":0.0,
                "median_true_match_rank":None,"p90_true_match_rank":None,"queries":0}
    recall={k:[] for k in ks}; reciprocal=[]; ndcg={k:[] for k in (5,10)}; true_ranks=[]
    for items in eligible:
        total=sum(rel for _,rel in items); ranks=[i for i,(_,rel) in enumerate(items,1) if rel]
        true_ranks.extend(ranks); reciprocal.append(1/min(ranks))
        for k in ks: recall[k].append(sum(rel for _,rel in items[:k])/total)
        for k in ndcg:
            dcg=sum(rel/math.log2(i+1) for i,(_,rel) in enumerate(items[:k],1)); ideal=sum(1/math.log2(i+1) for i in range(1,min(total,k)+1))
            ndcg[k].append(dcg/ideal if ideal else 0)
    ordered=sorted(true_ranks)
    return {**{f"recall_at_{k}":sum(v)/len(v) for k,v in recall.items()},"mrr":sum(reciprocal)/len(reciprocal),
            "ndcg_at_5":sum(ndcg[5])/len(ndcg[5]),"ndcg_at_10":sum(ndcg[10])/len(ndcg[10]),
            "median_true_match_rank":ordered[len(ordered)//2],"p90_true_match_rank":ordered[math.ceil(.9*len(ordered))-1],
            "queries":len(eligible)}


def ensure_non_authoritative(items: Iterable[RankedCandidate]) -> None:
    for item in items:
        forbidden={"should_merge","merge_probability","identity_probability","same_entity_probability"}
        if forbidden & item.model_dump().keys(): raise ValueError("candidate ranking must remain non-authoritative")
