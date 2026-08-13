"""Freeze the reviewed subset from completed public-RFC pipeline groups."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from devtools.real_entity_resolution_benchmark import freeze

CURATED={
 "rfc_foundations":[("URI",["M001","M051","M055"],"ABBREVIATION_EXPANSION"),("HTTP",["M030","M036","M038","M044","M047"],"ABBREVIATION_EXPANSION"),("TLS",["M059","M071"],"EXACT_NAME"),("ServerHello",["M088","M091"],"EXACT_NAME")],
 "rfc_transport":[("endpoint",["M037","M044","M049","M053","M056","M069","M081","M086","M088","M093"],"CASE_ONLY"),("HMAC",["M079","M094"],"ABBREVIATION_EXPANSION"),("RFC2104",["M080","M095"],"EXACT_NAME")]
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 all_sources=json.loads(a.sources.read_text(encoding='utf-8')); completed_documents={'rfc_identity_01','rfc_identity_02','rfc_identity_03','rfc_identity_04'}
 sources=[row for row in all_sources if row['document_id'] in completed_documents]; inventory=[]; clusters=[]; pairs=[]; queries=[]; entities={}
 for group,defs in CURATED.items():
  data=json.loads((a.root/'graphs'/group/'entity_embedding_inputs.json').read_text(encoding='utf-8'))
  by={x['local_entity_id']:x for x in data['inputs']}; inventory.append({'run_id':'benchmark_pipeline_run_v2','graph_group':group,'document_count':2,'local_entity_count':len(by),'status':'completed'});
  for label,ids,category in defs:
   cid=f"I{len(clusters)+1:03d}"; members=[]
   for eid in ids:
    x=by[eid]; doc=x['evidence_span_ids'][0].split(':')[0]; key=f'{group}:{eid}'; entities[key]=(group,x,doc); members.append({'entity_id':key,'canonical_name':x['name'],'entity_type':x['type'],'graph_group':group,'document_id':doc,'evidence_refs':x['evidence_span_ids']})
   clusters.append({'identity_cluster_id':cid,'canonical_concept':label,'members':members})
   for i in range(len(members)):
    for j in range(i+1,len(members)):
     l,r=members[i],members[j]; pairs.append({'left_entity_id':l['entity_id'],'right_entity_id':r['entity_id'],'label':'SAME_ENTITY','confidence':'HIGH','category':category,'label_reason':'same unambiguous RFC-defined named object','evidence_refs':l['evidence_refs']+r['evidence_refs'],'cross_document':l['document_id']!=r['document_id']})
   for q in members:
    pool=[]
    for c in members:
     if c!=q: pool.append({'candidate_entity_id':c['entity_id'],'candidate_document_id':c['document_id'],'expected':'SAME_ENTITY','category':category,'label_reason':'source-grounded cluster','evidence_refs':c['evidence_refs']})
    queries.append({'query_id':f'Q{len(queries)+1:03d}','identity_cluster_id':cid,'graph_group':group,'query_entity_id':q['entity_id'],'query_document_id':q['document_id'],'candidate_pool':pool})
 # Real hard negatives: distinct named clusters within each group; no semantic taxonomy inference.
 for i,left in enumerate(clusters):
  for right in clusters[i+1:]:
   l=left['members'][0];r=right['members'][0]
   if l['graph_group']==r['graph_group']:
    pairs.append({'left_entity_id':l['entity_id'],'right_entity_id':r['entity_id'],'label':'DIFFERENT_ENTITY','confidence':'HIGH','category':'RELATED_BUT_DISTINCT','label_reason':'RFC defines distinct named protocol objects','evidence_refs':l['evidence_refs']+r['evidence_refs']})
 result=freeze(a.output,sources,inventory,clusters,pairs,[],queries,[],[])
 review={'clusters':clusters,'pairs':pairs,'scores_included':False};(a.output/'review_packet.json').write_text(json.dumps(review,indent=2)+"\n",encoding='utf-8')
 lines=['# Entity Resolution Review Packet','','No model scores or resolver decisions are included.']
 for c in clusters:
  lines += ['',f"## {c['identity_cluster_id']}: {c['canonical_concept']}"]+[f"- {m['canonical_name']} ({m['entity_type']}) — {', '.join(m['evidence_refs'])}" for m in c['members']]
 (a.output/'review_packet.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 s=result['statistics'];q=result['quality'];m=result['manifest']; report=a.output.parent/'REAL_ENTITY_RESOLUTION_BENCHMARK_REPORT.md'
 report.write_text(f"# Real Entity Resolution Benchmark Report\n\n**ENTITY RESOLUTION BENCHMARK INSUFFICIENT**\n\nThe frozen real benchmark has {s['unique_positive_queries']} source-grounded positive queries, below 50; {s['negative_pairs']} hard negatives, below 100; and non-exact share {s['non_exact_positive_share']:.1%}. Stage B was not run.\n\nCompleted lineage covers 4 public RFC excerpt documents, 2 multi-document graph groups, {s['total_extracted_entities']} pre-resolution entities, one extraction call per span, and exact JSON/Neo4j parity. A third group was excluded because one span failed extraction.\n\nBenchmark version `{m['benchmark_version']}`, hash `{m['benchmark_hash']}`, split hash `{m['split_hash']}`.\n\nNext action: add real sources whose pipeline extraction produces separate full-name and short-form local entities, then conduct human review until at least 50 positive queries and 100 hard negatives pass the gate.\n",encoding='utf-8')
 print(json.dumps({'statistics':s,'quality':q,'manifest':m},indent=2));return 0
if __name__=='__main__': raise SystemExit(main())
