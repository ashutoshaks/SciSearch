import json
import jsonlines

class Paper:
    def __init__(self, paper_id, metadata, title, abstract, pred_labels_truncated, pred_labels):
        self.paper_id = paper_id
        self.metadata = metadata
        self.title = title
        self.abstract = abstract
        self.pred_labels_truncated = pred_labels_truncated
        self.pred_labels = pred_labels

papers = dict()
with jsonlines.open('../data/abstracts-csfcube-preds.jsonl') as reader:
    for obj in reader:
        papers[obj['paper_id']] = Paper(obj['paper_id'], obj['metadata'], obj['title'], obj['abstract'], obj['pred_labels_truncated'], obj['pred_labels'])

f = open('../data/test-pid2anns-csfcube-background.json')
q_background = json.load(f)
q_background_ids = list(q_background.keys())
print(q_background_ids)

f = open('../data/test-pid2anns-csfcube-method.json')
q_method = json.load(f)
q_method_ids = list(q_method.keys())
print(q_method_ids)

f = open('../data/test-pid2anns-csfcube-result.json')
q_result = json.load(f)
q_result_ids = list(q_result.keys())
print(q_result_ids)

def facet_sentences(pid, facet):
    labels = {'background': ['background_label', 'objective_label'], 'method': ['method_label'], 'result': ['result_label']}
    paper = papers[pid]
    s = list()
    for i in range(len(paper.pred_labels)):
        if paper.pred_labels[i] in labels[facet]:
            s.append(paper.abstract[i])
    return s

print(facet_sentences('10014168', 'result'))