def replace_token(x, representation_style, tokenizer):
    # x: pandas row
    # 문자열(x['sentence']) -> 리스트로 변환 후 삭제하고 해당 인덱스에 삽입
    x_sentence = x['sentence']

    # ['baseline', 'klue', 'matching_the_blank', 'punct', 'typed_marker', 'typed_punct_marker']

    if representation_style == 'klue':
        entity_markers = {'subj_s' : '<subj>', 'subj_e' : '</subj>', 'obj_s' : '<obj>', 'obj_e' : '</obj>'}
    elif representation_style == 'matching_the_blank':
        entity_markers = {'subj_s' : '[E1]', 'subj_e' : '</E1>', 'obj_s' : '<E2>', 'obj_e' : '</E2>'}
    elif representation_style == 'punct':
        entity_markers = {'subj_s' : '@', 'subj_e' : '@', 'obj_s' : '#', 'obj_e' : '#'}
        

    subj_s = x['subject_entity']['start_idx']
    subj_e = x['subject_entity']['end_idx']
    obj_s = x['object_entity']['start_idx']
    obj_e = x['object_entity']['end_idx']

    subj_type = x['subject_entity']['type']
    obj_type = x['object_entity']['type']

    subj_word = x['subject_entity']['word']
    obj_word = x['object_entity']['word']

    tmp = []

    # entity type 포함
    if representation_style == 'typed_marker':
        # [CLS]〈Something〉는 <o:PER>조지 해리슨</o:PER>이 쓰고 <s:ORG>비틀즈</s:ORG>가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        if subj_s < obj_s: # subj 가 먼저 나올 때
            tmp.extend([
                        x_sentence[:subj_s],
                        f'<s:{subj_type}>{subj_word}</s:{subj_type}>',
                        x_sentence[subj_e+1:obj_s], 
                        f'<o:{obj_type}>{obj_word}</o:{obj_type}>',
                        x_sentence[obj_e+1:]
                    ])
        
        elif subj_s > obj_s: # obj 가 먼저 나올 때
            tmp.extend([
                        x_sentence[:obj_s],
                        f'<o:{obj_type}>{obj_word}</o:{obj_type}>',
                        x_sentence[obj_e+1:subj_s], 
                        f'<s:{subj_type}>{subj_word}</s:{subj_type}>',
                        x_sentence[subj_e+1:]
                    ])
        else:
            raise ValueError("subj-obj overlapped")
    

    elif representation_style == 'typed_punct_marker':
        # [CLS]〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        if subj_s < obj_s: # subj 가 먼저 나올 때
            tmp = f'{x_sentence[:subj_s]}@*{subj_type}*{subj_word}@{x_sentence[subj_e+1:obj_s]}#^{obj_type}^{obj_word}#{x_sentence[obj_e+1:]}'
        
        elif subj_s > obj_s: # obj 가 먼저 나올 때
            tmp = f'{x_sentence[:obj_s]}#^{obj_type}^{obj_word}#{x_sentence[obj_e+1:subj_s]}@*{subj_type}*{subj_word}@{x_sentence[subj_e+1:]}'
        
        if subj_s < obj_s: # subj 가 먼저 나올 때
            tmp.extend([
                        x_sentence[:subj_s],
                        f'@*{subj_type}*{subj_word}@',
                        x_sentence[subj_e+1:obj_s],
                        f'#^{obj_type}^{obj_word}#',
                        x_sentence[obj_e+1:]
                    ])
        
        elif subj_s > obj_s: # obj 가 먼저 나올 때
            tmp.extend([
                        x_sentence[:obj_s],
                        f'#^{obj_type}^{obj_word}#',
                        x_sentence[obj_e+1:subj_s],
                        f'@*{subj_type}*{subj_word}@',
                        x_sentence[subj_e+1:]
                    ])
        else:
            raise ValueError("subj-obj overlapped")


    # entity type 불포함
    else:
        # f-string 말고 그냥 더하기로..
        if subj_s < obj_s: # subj 가 먼저 나올 때
            tmp.extend([
                x_sentence[:subj_s],
                entity_markers['subj_s'] + subj_word + entity_markers['subj_e'],
                x_sentence[subj_e+1:obj_s],
                entity_markers['obj_s'] + obj_word + entity_markers['obj_e'],
                x_sentence[obj_e+1:]
            ])

        elif subj_s > obj_s: # obj 가 먼저 나올 때

            tmp.extend([
                x_sentence[:obj_s],
                entity_markers['obj_s'] + obj_word + entity_markers['obj_e'],
                x_sentence[obj_e+1:subj_s],
                entity_markers['subj_s'] + subj_word + entity_markers['subj_e'],
                x_sentence[subj_e+1:]
            ])

        else:
            raise ValueError("subj-obj overlapped")

    # tokenized sentence, ss, os 반환하기
    # subject 시작 위치 (토큰 단위 계산)
    ss = len(tokenizer(tmp[0], add_special_tokens=False)['input_ids']) + 1
    os = ss + len(tokenizer(tmp[1], add_special_tokens=False)['input_ids']) + len(tokenizer(tmp[2], add_special_tokens=False)['input_ids'])
    
    if subj_s > obj_s:
        ss, os = os, ss
    
    text = "".join(tmp)

    outputs = tokenizer(text, 
                        return_tensors="pt", 
                        padding="max_length", 
                        truncation=True, 
                        max_length=256, 
                        add_special_tokens=True)

    return outputs['input_ids'], ss, os