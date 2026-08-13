"""Generate non-authoritative pair proposals from accepted extraction artifacts."""
from __future__ import annotations
import json,re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

TOKENS=re.compile(r"\w+",re.UNICODE)

def accepted_partial_entities(root:Path,document_ids:list[str],graph_group:str)->list[dict[str,Any]]:
 out=[]
 for doc in document_ids:
  units={row['document_local_order']:row for row in json.loads((root/'documents'/doc/'units.json').read_text(encoding='utf-8'))}
  for folder in sorted((root/'documents'/doc/'units').glob('span_*')):
   result=folder/'span_extraction_results.json'
   if not result.exists(): continue
   order=int(folder.name.split('_')[-1]); meta=units[order]; rows=json.loads(result.read_text(encoding='utf-8'))
   if not rows: continue
   for entity in rows[0].get('entities',[]):
    out.append({'entity_id':f"{graph_group}:{doc}:S{order:03d}:{entity['local_id']}",'canonical_name':entity['name'],
      'entity_type':entity['type'],'graph_group':graph_group,'document_id':doc,
      'evidence_refs':[f"{doc}:S{order:03d}"],'source_start':meta['source_start'],'source_end':meta['source_end'],
      'source_coverage':'PARTIAL_SOURCE_COVERAGE'})
 return out

def propose(entities:list[dict[str,Any]])->list[dict[str,Any]]:
 rows=[]
 for i,left in enumerate(entities):
  a=' '.join(TOKENS.findall(left['canonical_name'].casefold()))
  for right in entities[i+1:]:
   if left['graph_group']!=right['graph_group']: continue
   b=' '.join(TOKENS.findall(right['canonical_name'].casefold())); signals=[]
   if a==b: signals.append('EXACT_OR_CASE')
   if a in b or b in a: signals.append('NAME_CONTAINMENT')
   if left['entity_type'].casefold()==right['entity_type'].casefold(): signals.append('SAME_TYPE')
   if left['evidence_refs']==right['evidence_refs']: signals.append('SAME_SPAN')
   elif abs(int(left['evidence_refs'][0].split('S')[-1])-int(right['evidence_refs'][0].split('S')[-1]))==1 and left['document_id']==right['document_id']: signals.append('ADJACENT_SPAN')
   similarity=SequenceMatcher(None,a,b).ratio()
   if similarity>=.55: signals.append('LEXICAL_SIMILARITY')
   if signals and (similarity>=.55 or 'SAME_SPAN' in signals or 'NAME_CONTAINMENT' in signals):
    rows.append({'proposal_id':f'P{len(rows)+1:05d}','left_entity_id':left['entity_id'],'right_entity_id':right['entity_id'],
      'signals':signals,'lexical_similarity':round(similarity,6),'proposed_label':None,
      'review_status':'REQUIRES_HUMAN_SOURCE_REVIEW','evidence_refs':left['evidence_refs']+right['evidence_refs']})
 return rows
