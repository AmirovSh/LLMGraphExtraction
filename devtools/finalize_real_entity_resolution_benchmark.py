"""Freeze the reviewed subset from completed public-RFC pipeline groups."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from devtools.real_entity_resolution_benchmark import freeze
from devtools.entity_resolution_review_proposals import accepted_partial_entities,propose

CURATED={
 "rfc_foundations":[("URI",["M001","M051","M055"],"ABBREVIATION_EXPANSION"),("HTTP",["M030","M036","M038","M044","M047"],"ABBREVIATION_EXPANSION"),("TLS",["M059","M071"],"EXACT_NAME"),("ServerHello",["M088","M091"],"EXACT_NAME")],
 "rfc_transport":[("endpoint",["M037","M044","M049","M053","M056","M069","M081","M086","M088","M093"],"CASE_ONLY"),("HMAC",["M079","M094"],"ABBREVIATION_EXPANSION"),("RFC2104",["M080","M095"],"EXACT_NAME")]
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 all_sources=json.loads(a.sources.read_text(encoding='utf-8')); completed_documents={'rfc_identity_01','rfc_identity_02','rfc_identity_03','rfc_identity_04','rfc_identity_05','rfc_identity_06'}
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
 partial=accepted_partial_entities(a.root,['rfc_identity_05','rfc_identity_06'],'rfc_security_partial'); by_partial={x['entity_id']:x for x in partial}
 partial_defs=[
  ('TCP',['rfc_security_partial:rfc_identity_05:S003:S001_M002','rfc_security_partial:rfc_identity_05:S004:S001_M001','rfc_security_partial:rfc_identity_05:S006:S001_M001'],'ABBREVIATION_EXPANSION'),
  ('Authentication Service',['rfc_security_partial:rfc_identity_06:S002:S001_M001','rfc_security_partial:rfc_identity_06:S004:S001_M002'],'ABBREVIATION_EXPANSION'),
  ('Delivery Service',['rfc_security_partial:rfc_identity_06:S003:S001_M001','rfc_security_partial:rfc_identity_06:S004:S001_M003','rfc_security_partial:rfc_identity_06:S006:S001_M005'],'ABBREVIATION_EXPANSION'),
  ('MLS',['rfc_security_partial:rfc_identity_06:S004:S001_M001','rfc_security_partial:rfc_identity_06:S006:S001_M001','rfc_security_partial:rfc_identity_06:S008:S001_M001'],'EXACT_NAME'),
  ('forward secrecy',['rfc_security_partial:rfc_identity_05:S017:S001_M004','rfc_security_partial:rfc_identity_06:S014:S001_M002'],'EXACT_NAME'),
  ('post-compromise security',['rfc_security_partial:rfc_identity_05:S017:S001_M005','rfc_security_partial:rfc_identity_06:S014:S001_M003'],'EXACT_NAME'),
  ('authorization server',['rfc_security_partial:rfc_identity_06:S017:S001_M003','rfc_security_partial:rfc_identity_06:S020:S001_M001'],'EXACT_NAME'),
  ('sequence number',['rfc_security_partial:rfc_identity_05:S010:S001_M001','rfc_security_partial:rfc_identity_05:S011:S001_M002'],'EXACT_NAME'),
  ('first data octet',['rfc_security_partial:rfc_identity_05:S010:S001_M002','rfc_security_partial:rfc_identity_05:S011:S001_M004'],'EXACT_NAME'),
  ('client',['rfc_security_partial:rfc_identity_05:S020:S001_M006','rfc_security_partial:rfc_identity_05:S021:S001_M003'],'EXACT_NAME')]
 partial_defs.append(('key',['rfc_security_partial:rfc_identity_05:S016:S001_M001','rfc_security_partial:rfc_identity_05:S016:S001_M005'],'SINGULAR_PLURAL'))
 singleton_ids=['rfc_security_partial:rfc_identity_05:S007:S001_M004','rfc_security_partial:rfc_identity_05:S011:S001_M003','rfc_security_partial:rfc_identity_05:S020:S001_M005','rfc_security_partial:rfc_identity_06:S008:S001_M002','rfc_security_partial:rfc_identity_06:S009:S001_M003','rfc_security_partial:rfc_identity_06:S012:S001_M005','rfc_security_partial:rfc_identity_06:S012:S001_M006','rfc_security_partial:rfc_identity_06:S021:S001_M002']
 partial_defs += [(by_partial[x]['canonical_name'],[x],'SINGLETON') for x in singleton_ids]
 for label,ids,category in partial_defs:
  cid=f"I{len(clusters)+1:03d}"; members=[]
  for eid in ids:
   x=by_partial[eid]; members.append({'entity_id':eid,'canonical_name':x['canonical_name'],'entity_type':x['entity_type'],'graph_group':x['graph_group'],'document_id':x['document_id'],'evidence_refs':x['evidence_refs'],'source_coverage':'PARTIAL_SOURCE_COVERAGE'})
  clusters.append({'identity_cluster_id':cid,'canonical_concept':label,'members':members,'source_coverage':'PARTIAL_SOURCE_COVERAGE'})
  for i in range(len(members)):
   for j in range(i+1,len(members)):
    l,r=members[i],members[j];pairs.append({'left_entity_id':l['entity_id'],'right_entity_id':r['entity_id'],'label':'SAME_ENTITY','confidence':'HIGH','category':category,'label_reason':'reviewed explicit definition or unambiguous reuse in accepted RFC spans','evidence_refs':l['evidence_refs']+r['evidence_refs'],'cross_document':l['document_id']!=r['document_id'],'source_coverage':'PARTIAL_SOURCE_COVERAGE'})
  if len(members)>1:
   for q in members: queries.append({'query_id':f'Q{len(queries)+1:03d}','identity_cluster_id':cid,'graph_group':q['graph_group'],'query_entity_id':q['entity_id'],'query_document_id':q['document_id'],'candidate_pool':[{'candidate_entity_id':c['entity_id'],'candidate_document_id':c['document_id'],'expected':'SAME_ENTITY','category':category,'label_reason':'reviewed cluster','evidence_refs':c['evidence_refs']} for c in members if c!=q]})
 inventory.append({'run_id':'benchmark_pipeline_run_v2','graph_group':'rfc_security_partial','document_count':2,'local_entity_count':len(partial),'status':'PARTIAL_SOURCE_COVERAGE','accepted_spans':51,'failed_spans':1})
 # Reviewed cross-cluster negatives: proposal signals never assign these labels.
 for i,left in enumerate(clusters):
  for right in clusters[i+1:]:
   l=left['members'][0];r=right['members'][0]
   if l['graph_group']==r['graph_group']:
    pairs.append({'left_entity_id':l['entity_id'],'right_entity_id':r['entity_id'],'label':'DIFFERENT_ENTITY','confidence':'HIGH','category':'RELATED_BUT_DISTINCT','label_reason':'human/source review: RFC defines separate named objects or concepts','evidence_refs':l['evidence_refs']+r['evidence_refs']})
 # Add reviewed hard distractors to query pools without duplicating semantic evidence.
 member_cluster={m['entity_id']:c for c in clusters for m in c['members']}
 for q in queries:
  candidates={x['candidate_entity_id'] for x in q['candidate_pool']}; qc=member_cluster[q['query_entity_id']]
  for c in clusters:
   if c is qc or c['members'][0]['graph_group']!=q['graph_group']: continue
   m=c['members'][0]
   if m['entity_id'] not in candidates:q['candidate_pool'].append({'candidate_entity_id':m['entity_id'],'candidate_document_id':m['document_id'],'expected':'DIFFERENT_ENTITY','category':'RELATED_BUT_DISTINCT','label_reason':'reviewed distinct cluster','evidence_refs':m['evidence_refs']})
 result=freeze(a.output,sources,inventory,clusters,pairs,[],queries,[],[])
 proposals=propose([m for c in clusters for m in c['members']]); (a.output/'review_proposals.json').write_text(json.dumps(proposals,indent=2)+"\n",encoding='utf-8')
 review={'clusters':clusters,'pairs':pairs,'scores_included':False,'proposal_labels_are_non_authoritative':True};(a.output/'review_packet.json').write_text(json.dumps(review,indent=2)+"\n",encoding='utf-8')
 lines=['# Entity Resolution Review Packet','','No model scores or resolver decisions are included.']
 for c in clusters:
  lines += ['',f"## {c['identity_cluster_id']}: {c['canonical_concept']}"]+[f"- {m['canonical_name']} ({m['entity_type']}) — {', '.join(m['evidence_refs'])}" for m in c['members']]
 (a.output/'review_packet.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 s=result['statistics'];q=result['quality'];m=result['manifest']; report=a.output.parent/'REAL_ENTITY_RESOLUTION_BENCHMARK_REPORT.md'; verdict='PASS' if q['status']=='PASS' else 'INSUFFICIENT'
 report.write_text(f"# Real Entity Resolution Benchmark Report\n\n**ENTITY RESOLUTION BENCHMARK {verdict}**\n\n## Sources and selection\n\nSix verbatim public RFC excerpt documents were selected as three related multi-document sets. HTTP/TLS and HTTP2/QUIC were chosen for high acronym and related-component density. TCP/MLS/OAuth/MIME was chosen for explicit full-name/short-form definitions and dense, distinct protocol objects. Expected alias density and expected hard-negative density were recorded as HIGH for each set in the review rationale.\n\n## Frozen dataset\n\n- extracted entities: {s['total_extracted_entities']}\n- identity clusters: {s['identity_clusters']}\n- unique positive queries: {s['unique_positive_queries']}\n- reviewed hard negatives: {s['negative_pairs']}\n- ambiguous: {s['ambiguous_pairs']}\n- cross-document positive pairs: {s['cross_document_positives']}\n- non-exact positive share: {s['non_exact_positive_share']:.1%}\n- related-but-distinct negatives: {s['related_but_distinct_negatives']}\n\nThe third graph group is explicitly `PARTIAL_SOURCE_COVERAGE`: 51 accepted spans contribute extraction-grounded entities; one failed span and the unpublished group FinalGraph contribute nothing. Proposal signals never assign labels. Reversed pairs and repeated runs are not independent evidence.\n\nQuality gate: **{q['status']}**. Benchmark version `{m['benchmark_version']}`, hash `{m['benchmark_hash']}`, split hash `{m['split_hash']}`.\n",encoding='utf-8')
 print(json.dumps({'statistics':s,'quality':q,'manifest':m},indent=2));return 0
if __name__=='__main__': raise SystemExit(main())
