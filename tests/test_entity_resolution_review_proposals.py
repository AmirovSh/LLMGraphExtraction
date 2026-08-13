from devtools.entity_resolution_review_proposals import propose

def test_signals_only_propose_and_never_label():
 e=[{'entity_id':'a','canonical_name':'Service Record','entity_type':'artifact','graph_group':'g','document_id':'d','evidence_refs':['d:S001']},
    {'entity_id':'b','canonical_name':'Service','entity_type':'component','graph_group':'g','document_id':'d','evidence_refs':['d:S002']}]
 rows=propose(e)
 assert rows and rows[0]['proposed_label'] is None
 assert rows[0]['review_status']=='REQUIRES_HUMAN_SOURCE_REVIEW'

def test_cross_group_pairs_are_never_proposed():
 e=[{'entity_id':x,'canonical_name':'Same','entity_type':'x','graph_group':x,'document_id':x,'evidence_refs':[f'{x}:S001']} for x in ('a','b')]
 assert propose(e)==[]
