import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0 # no_relation은 평가에서 제외, micro : 샘플 개수를 고려한 평균

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    # probs shape : (B, 30)
    # labels shape : (B, )

    labels = np.eye(30)[labels] # label을 one-hot vector로 변환 # shape : (B , 30)

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel() # class c : (B, 1) -> (B, )
        preds_c = probs.take([c], axis=1).ravel() # class c : (B, 1) -> (B, )
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(logits, labels):
  """ validation을 위한 metrics function """
  # logits shape : (B, 30)
  # labels shape : (B, )

  labels = labels.cpu().numpy() 
  preds = logits.argmax(-1).cpu().numpy() # shape : (B, 30) -> (B, )
  probs = logits.softmax(-1).cpu().numpy()

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  # auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      # 'auprc' : auprc,
      'accuracy': acc,
  }