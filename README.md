# Level2-klue-nlp-06: Relation Extraction

## 📌 대회 설명
## Relation Extraction (2024.01.03 ~ 2024.01.18)
<aside>
  
💡 문장 내에서 **문장의 단어 (Entity)** 에 대한 **속성과 관계를 예측**하는 문제

</aside>

- 프로젝트 기간 (2024.01.03 ~ 2024.01.18)
- 학습/평가 데이터
    - train.csv: 총 32470개
    - test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현)
    ![image](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/0995baf4-9900-47d4-a93b-caf8fc122a94)

- 평가 지표 : KLUE-RE evaluation metric을 그대로 재현
    - 1) no_relation class를 제외한 **micro F1 score**
        
        2) 모든 class에 대한 area under the precision-recall curve (AUPRC)
<br>

## 📌 Learder Board
### **🥈Private Leader Board(2위)**
![image](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/625ccef3-229a-463f-9072-87b585f8d9ef)
### **🥉Public Leader Board(3위)**
![image](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/80e0eae9-dc69-419d-a511-f879745e933c)

<br>

## 📌 팀 소개

* **Team명** : 찐친이 되어줘 [NLP 6조]

|김재현|서동해|송민환|장수정|황예원|황재훈|
|:--:|:--:|:--:|:--:|:--:|:--:|
|![재현](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/fa007f29-007b-42c0-bb1a-f95176ad7d93)|![동해-PhotoRoom png-PhotoRoom](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/7ba86ba4-cd7a-4366-97aa-7669e7994a78)|![민환](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/a3614eb6-4757-4390-9196-f82a455b4418)|![수정](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/39b8b55c-d1d8-4125-bbf2-11a695bcbc23)|![예원-PhotoRoom png-PhotoRoom](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/46ab92c3-e6cc-455a-b9c3-a225c8730048)|![재훈-removebg-preview](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/5d8cf554-d59a-44fa-802d-38bd66111263)|
|[Github](https://github.com/finn-sharp)|[Github](https://github.com/DonghaeSuh)|[Github](https://github.com/codestudy25)|[Github](https://github.com/jo9392)|[Github](https://github.com/yeowonh)|[Github](https://github.com/iloveonsen)|
|[Mail](penguin-klg@jnu.ac.kr)|[Mail](donghaesuh2@gmail.com)|[Mail](meenham_song@naver.com)|[Mail](jo23892389@gmail.com)|[Mail](yeowonh@sju.ac.kr)|[Mail](mgs05144@gmail.com)|

<br>

## 📌 프로젝트 역할
| 팀원명 | 역할 |
| --- | --- |
| **김재현(T6036)** |  |
| **서동해(T6077)** | Pytorch Lightning 기반 베이스라인 코드 작성, 라벨 분석을 통한 Train_Dev 데이터 분리, 개체명 인식 모델을 활용해 오답률 기반 라벨 데이터 증강, TAPT 기반 추가 사전학습 |
| **송민환(T6086)** |  |
| **장수정(T6148)** | 다양한 기법 탐색(ideation), 리팩토링, focal loss 추가 및 성능 비교 실험, 앙상블 실험에 참여 |
| **황예원(T6191)** | RoBERTa-large를 기반으로 Entity marker, hidden token vector, mean pooling 에 대한 실험 및 성능 비교 진행, 추가 프롬프트 실험과 앙상블 실험에 참여. |
| **황재훈(T6193)** |  |

<br>

## 📌 실험한 것들

|**Process**|**What we did**|
|:--:|--|
|**EDA**|`라벨 분포 분석`, `Baseline 모델 예측과 실제값 차이 분석`, `오답률 분석`|
|**Preprocessing**| `Entity Marker`, `Semantic Typing` , `이상치 및 중복 데이터 제거`|
|**Augmentation**|`TAPT MASK Prediction`, `NER 모델 기반 데이터 증강`|
|**Modeling**|`Entity Embedding`, `Mean Pooling`, `K-epic`, `TAPT`|
|**Experiment Model**|`klue/roberta-large`, `wooy0ng/korquad1-klue-roberta-large`, `studio-ousia/mluke-large-lite`, `hfl/cino-large-v2`, `google/rembert`, `FacebookAI/xlm-roberta-large`|
|**Mornitoring**| `Wandb`|
|**Ensemble**|`weight voting`, `soft voting`, `hard voting`|

<br>

## 📌 코드 구조
```
┣ 📂.github
┃ ┣ 📂ISSUE_TEMPLATE
┃ ┃ ┣ bug_report.md
┃ ┃ ┗ feature_request.md
┃ ┣ .keep
┃ ┗ PULL_REQUEST_TEMPLATE.md
┣ 📂augmentation
┃ ┣ mlm_augmentation.py
┃ ┗ roberta_pretrain.py
┣ 📂configs
┃ ┣ augment_config.json
┃ ┣ augment1_roberta-large_config.json
┃ ┣ auto_config.json
┃ ┣ base_config.json
┃ ┣ cino_type_entity_config.json
┃ ┣ entity_config.json
┃ ┣ focal_test.json
┃ ┣ korquad1-klue-roberta-large_type_entity_config.json
┃ ┣ korquad1-klue-roberta-large_type_prompt_config.json
┃ ┣ mlm_augmentation_config.json
┃ ┣ mluke-large-lite_type_entity_config.json
┃ ┣ pretrained_roberta-large_config.json
┃ ┣ pretrained_roberta-large_endtoken_config.json
┃ ┣ pretrained_roberta-large_pooling_config.json
┃ ┣ rembert-typed_punct_prompt_config.json
┃ ┣ roberta-large_config.json
┃ ┗ roberta-large_entity_config.json
┣ 📂data_analysis
┃ ┣ 📂ner_augmentation
┃ ┃ ┣ kor_ner_test.ipynb
┃ ┃ ┣ label.py
┃ ┃ ┣ ner_augment.ipynb
┃ ┃ ┗ ners.csv
┃ ┣ data_checking_ywh.ipynb
┃ ┣ data_description.txt
┃ ┣ dhs_confusion_matrix.ipynb
┃ ┣ dhs_data_manipulate.ipynb
┃ ┣ dhs_error_rate.ipynb
┃ ┣ dhs_label_analysis.ipynb
┃ ┗ mhs.ipynb
┣ 📂models
┃ ┣ base_model.py
┃ ┣ entity_marker_endtoken_model.py
┃ ┣ entity_marker_model.py
┃ ┣ entity_marker_pooling_model.py
┃ ┣ entity_model.py
┃ ┣ merged_test_data.csv
┃ ┣ prompt_marker_model.py
┃ ┗ typing_model.py
┣ 📂prediction
┃ ┗ sample_submission.csv
┣ 📂utils
┃ ┣ dict_label_to_num.pkl
┃ ┣ dict_num_to_label.pkl
┃ ┣ ensemble.ipynb
┃ ┣ loss.py
┃ ┣ replace_representation_tokens.py
┃ ┣ seed.py
┃ ┗ utils.py
┣ .gitignore
┣ auto_inference.py
┣ auto_train.py
┣ dataloader_endtoken.py
┣ dataloader_prompt.py
┣ dataloader_prompt2.py
┣ dataloader.py
┣ entity_dataloader.py
┣ entity_inference.py
┣ entity_train.py
┣ inference.py
┣ metric.py
┣ README.md
┣ requirements.txt
┣ test_copy.ipynb
┣ train.py
┗ typing_dataloader.py
```
