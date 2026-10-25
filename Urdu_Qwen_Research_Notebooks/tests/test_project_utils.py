import json
from pathlib import Path
from collections import Counter
import pytest
from project_utils import (DEFAULT_LABELS, PAPER_LABELS, normalize_label, stratified_prefix_samples,
                           validate_split, encode_supervised, evaluate_predictions, messages_for,
                           stable_id, file_hash, write_json, read_json)


class ToyChatTokenizer:
    """Fake tokenizer with the exact same prefix property expected of an instruct LLM."""
    def encode(self, text, add_special_tokens=False):
        return [ord(x) for x in text]
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(x) for x in ids)
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
        if add_generation_prompt:
            rendered += '<assistant>'
        return [ord(ch) for ch in rendered] if tokenize else rendered


def make_rows():
    return [dict(example_id=stable_id('toy','ur','train',i),lang='ur',split='train',
                 label='sports' if i%2 else 'politics',text=f'unique sample {i}',index_id=str(i)) for i in range(20)]


def test_strict_labels():
    assert normalize_label('  SPORTS.  ', DEFAULT_LABELS)=='sports'
    assert normalize_label('science/technology', DEFAULT_LABELS)=='science/technology'
    assert normalize_label('sports and politics', DEFAULT_LABELS) is None
    assert normalize_label('Here is the label: sports', DEFAULT_LABELS) is None
    assert normalize_label('science', DEFAULT_LABELS) is None
    assert normalize_label('world', DEFAULT_LABELS) is None
    assert len(PAPER_LABELS)==8 and len(DEFAULT_LABELS)==7


def test_no_replacement_and_nested_sampling():
    rows=make_rows()
    result=stratified_prefix_samples(rows,[2,5,11],42)
    assert len(result[2]['rows'])==4
    assert len(result[5]['rows'])==10
    assert not result[11]['available']
    ids2={r['example_id'] for r in result[2]['rows']}
    ids5={r['example_id'] for r in result[5]['rows']}
    assert ids2 <= ids5
    assert result[5]['rows']==stratified_prefix_samples(rows,[5],42)[5]['rows']
    assert Counter(r['label'] for r in result[5]['rows']) == {'politics':5,'sports':5}


def test_cross_split_text_leakage():
    a=make_rows()
    b=dict(a[0]); b['example_id']='new'; b['split']='test'; b['index_id']='100'
    assert any('text across splits' in x for x in validate_split(a+[b],['politics','sports']))


def test_cross_split_id_leakage():
    a=make_rows()
    b=dict(a[0]);b['example_id']='new';b['split']='test'; b['text']='different'
    assert any('index_id across splits' in x for x in validate_split(a+[b],['politics','sports']))



class BoundaryMergingTokenizer(ToyChatTokenizer):
    """Simulate a tokenizer that merges a token across the prompt/answer boundary."""
    def encode(self, text, add_special_tokens=False):
        target = '<assistant>sports'
        if target not in text:
            return super().encode(text, add_special_tokens=add_special_tokens)
        before, after = text.split(target, 1)
        return super().encode(before) + [999999] + super().encode(after)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = super().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.encode(text, add_special_tokens=False) if tokenize else text


def test_bpe_boundary_merging_preserves_inference_prompt_and_answer_mask():
    tk = BoundaryMergingTokenizer()
    prompt = messages_for('urdu test', ['sports', 'politics'], 'ur')
    prefix = tk.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    complete = tk.apply_chat_template(
        prompt + [{'role': 'assistant', 'content': 'sports'}],
        tokenize=True, add_generation_prompt=False
    )
    assert complete[:len(prefix)] != prefix  # Previous implementation crashed here.
    encoded = encode_supervised(tk, 'urdu test', 'sports', ['sports', 'politics'], 'ur', 2048)
    assert encoded['input_ids'][:len(prefix)] == prefix
    assert encoded['labels'][:len(prefix)] == [-100] * len(prefix)
    supervised = ''.join(chr(x) for x in encoded['labels'] if x != -100)
    assert supervised.startswith('sports') and supervised.endswith('</assistant>')
    assert len(encoded['input_ids']) == len(encoded['labels'])

def test_supervision_masks_prompt():
    tokenizer=ToyChatTokenizer()
    item=encode_supervised(tokenizer,'urdu test','sports',['sports','politics'],'ur',2048)
    assert item['labels'].count(-100)>0
    assert item['labels'][-1] != -100
    assert len(item['labels']) == len(item['input_ids'])
    assert item['labels'].count(-100) < len(item['labels'])


def test_overlong_is_rejected_not_silent():
    with pytest.raises(ValueError,match=r'Prompt\+target alone exceeds'):
        encode_supervised(ToyChatTokenizer(),'a'*50,'sports',['sports','politics'],'en',max_length=20)


def test_strict_prediction_alignment():
    rows=[dict(example_id='a',label='sports'),dict(example_id='b',label='politics')]
    pred=[dict(example_id='a',prediction='sports'), dict(example_id='b',prediction='garbage')]
    out=evaluate_predictions(rows,pred,['sports','politics'])
    assert out['accuracy']==0.5 and out['invalid_count']==1
    assert out['macro_f1']<1
    with pytest.raises(ValueError,match='mismatched'):
        evaluate_predictions(rows,[pred[0]],['sports','politics'])
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate_predictions(rows,[pred[0],pred[0]],['sports','politics'])


def test_no_test_demonstration_default():
    demos=make_rows()[:2]
    m=messages_for('other text',['sports','politics'],demonstrations=demos)
    assert len(m)==6
    assert m[1]['role']=='user' and m[2]['role']=='assistant'


def test_write_read_manifest(tmp_path):
    p=tmp_path/'sub'/'manifest.json'
    write_json(p,{'labels':['sports','politics'],'hi':'اردو'})
    assert read_json(p)['hi']=='اردو' and len(file_hash(p))==64


def test_collator_masks_padding_and_preserves_targets():
    import torch
    from project_utils import supervised_collator
    tok = type('Toy', (), {'pad_token_id':0, 'eos_token_id':2})()
    batch = supervised_collator(tok)([
        {'input_ids':[11,12,13], 'attention_mask':[1,1,1], 'labels':[-100,-100,13]},
        {'input_ids':[21,22], 'attention_mask':[1,1], 'labels':[-100,22]},
    ])
    assert batch['input_ids'].shape==(2,3)
    assert batch['attention_mask'][1].tolist()==[1,1,0]
    assert batch['labels'][1].tolist()==[-100,22,-100]


def test_inference_resume_and_metadata_protection(monkeypatch,tmp_path):
    import project_utils as m
    monkeypatch.setattr(m, 'run_generation', lambda model,tok,row,labels,lang,max_new_tokens:
                        dict(raw_prediction='sports',prediction='sports',valid=True,seconds=0.02))
    rows=[dict(example_id='a',label='sports',text='a'),dict(example_id='b',label='sports',text='b')]
    meta=dict(task='demo',lang='ur',split='test',dataset_sha256='abcdef', model_id='base',run_id='r1',
              method='lora',subset=False,decoding={'do_sample':False,'max_new_tokens':16})
    path=tmp_path/'predictions.csv'
    results=m.infer_resume(None,None,rows,['sports'],'ur',path,meta)
    assert len(results)==2 and path.exists()
    second=m.infer_resume(None,None,rows,['sports'],'ur',path,meta)
    assert len(second)==2
    bad=dict(meta);bad['model_id']='different'
    with pytest.raises(ValueError, match='different'):
        m.infer_resume(None,None,rows,['sports'],'ur',path,bad)


def test_training_truncates_source_but_not_answer():
    tk=ToyChatTokenizer()
    item=encode_supervised(tk,'a'*200,'sports',['sports','politics'],'en',max_length=350)
    assert item['source_truncated'] and item['removed_source_tokens']>0
    target=[i for i in item['labels'] if i!=-100]
    assert ''.join(chr(x) for x in target).startswith('sports')


# Offline checks for the exact 2026 Kaggle-native architecture (Transformers 5).
from project_utils import (
    training_arguments_kwargs, trainer_processing_arg, numeric_token_ids,
    supervised_collator, run_generation,
)
import inspect
import torch


def _fake_class_with_signature(keys):
    class Fake:
        def __init__(self, **kwargs):
            self.options = kwargs
    Fake.__signature__ = inspect.Signature([
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY)
        for k in sorted(keys)
    ])
    return Fake


def _supported_options():
    config = {
        'train_epochs': 2, 'per_device_batch_size': 1,
        'gradient_accumulation': 8,
        'learning_rate_lora': 2e-4, 'learning_rate_full': 2e-5,
        'optimizer': 'adafactor', 'max_grad_norm': 1.0,
    }
    opts = {
        'output_dir', 'num_train_epochs', 'max_steps', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'gradient_accumulation_steps', 'learning_rate',
        'optim', 'max_grad_norm', 'fp16', 'bf16', 'gradient_checkpointing',
        'gradient_checkpointing_kwargs', 'save_strategy', 'save_steps',
        'save_total_limit', 'logging_steps', 'dataloader_num_workers',
        'remove_unused_columns', 'report_to', 'seed', 'data_seed',
        'lr_scheduler_type', 'warmup_steps', 'eval_strategy',
        'train_sampling_strategy', 'ddp_find_unused_parameters',
    }
    return config, opts


def test_native5_training_arguments_lora(tmp_path):
    config, opts = _supported_options()
    version5 = _fake_class_with_signature(opts)
    result = training_arguments_kwargs(config, 'lora', tmp_path, 42, None, 2, True, True, version5)
    assert result['train_sampling_strategy'] == 'random'
    assert result['warmup_steps'] == 0.03
    assert result['gradient_checkpointing_kwargs'] == {'use_reentrant': False}
    assert result['fp16'] is True
    assert result['save_strategy'] == 'no'
    assert result['max_steps'] == 2
    assert not {'overwrite_output_dir', 'warmup_ratio', 'group_by_length', 'evaluation_strategy'} & set(result)
    assert version5(**result).options == result


def test_native5_training_arguments_full(tmp_path):
    config, opts = _supported_options()
    version5 = _fake_class_with_signature(opts)
    result = training_arguments_kwargs(config, 'full', tmp_path, 42, None, None, False, False, version5)
    assert result['learning_rate'] == 2e-5
    assert result['fp16'] is False
    assert result['save_strategy'] == 'steps'
    assert result['max_steps'] == -1
    assert version5(**result).options == result


def test_native5_api_mismatch_fails_before_training(tmp_path):
    config, opts = _supported_options()
    version4 = _fake_class_with_signature(opts - {'train_sampling_strategy'} | {'group_by_length'})
    with pytest.raises(RuntimeError, match='train_sampling_strategy'):
        training_arguments_kwargs(config,'lora',tmp_path,42,None,2,True,False,version4)


def test_unknown_method_fails(tmp_path):
    config, opts = _supported_options()
    version5 = _fake_class_with_signature(opts)
    with pytest.raises(ValueError, match='Unknown training method'):
        training_arguments_kwargs(config, 'test', tmp_path, 42, None, 2, True, False, version5)


def test_trainer_processing_class_native5():
    version5 = _fake_class_with_signature(['model','args','train_dataset','processing_class'])
    version4 = _fake_class_with_signature(['model','args','train_dataset','tokenizer'])
    marker = object()
    assert trainer_processing_arg(version5, marker) == {'processing_class':marker}
    with pytest.raises(RuntimeError, match='processing_class'):
        trainer_processing_arg(version4, marker)


def test_batch_encoding_mapping_is_flat_numeric():
    class FakeBatchEncoding(dict):
        pass
    ids = numeric_token_ids(FakeBatchEncoding({'input_ids': [[11,22,33]], 'attention_mask': [[1,1,1]]}))
    assert ids == [11,22,33]
    assert numeric_token_ids(torch.tensor([[5,6]])) == [5,6]
    with pytest.raises(ValueError, match='non-integer token'):
        numeric_token_ids(['input_ids','attention_mask',12])


def test_supervised_encoder_batchencoding_regression():
    class EncodingDictTokenizer(ToyChatTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            output = super().apply_chat_template(messages, tokenize=tokenize,
                                                add_generation_prompt=add_generation_prompt)
            if tokenize:
                return {'input_ids':output, 'attention_mask':[1]*len(output)}
            return output
    e = encode_supervised(EncodingDictTokenizer(), 'short source', 'sports',
                          ['sports','politics'], 'en', 1024)
    assert all(isinstance(i,int) for i in e['input_ids'])
    assert 'sports' in EncodingDictTokenizer().decode([x for x in e['labels'] if x != -100])


def test_collator_rejects_non_numeric_early():
    tok = type('Tokenizer',(),{'pad_token_id':0,'eos_token_id':2})()
    with pytest.raises(ValueError, match='non-integer token'):
        supervised_collator(tok)([{'input_ids':['input_ids',3],
                                    'attention_mask':[1,1], 'labels':[-100,3]}])


def test_inference_mapping_and_tensor(monkeypatch):
    class Model:
        def __init__(self): self.p = torch.nn.Parameter(torch.tensor([1.]))
        def parameters(self): return iter([self.p])
        def generate(self, input_ids, attention_mask, **kwargs):
            assert input_ids.shape == attention_mask.shape
            return torch.cat([input_ids,torch.tensor([[ord(x) for x in 'sports']],device=input_ids.device)],dim=1)
    class Tok(ToyChatTokenizer):
        pad_token_id = 0
        eos_token_id = 2
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_tensors=None):
            raw=super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
            if tokenize and return_tensors:
                return {'input_ids':torch.tensor([raw]), 'attention_mask':torch.ones((1,len(raw)),dtype=torch.long)}
            return raw
    row={'text':'test','label':'sports'}
    result=run_generation(Model(),Tok(),row,['sports','politics'])
    assert result['prediction']=='sports' and result['valid']


def test_requirements_record_preinstalled_versions():
    root=Path(__file__).resolve().parents[1]
    content=(root/'requirements.txt').read_text()
    assert 'transformers==5.16.1' in content
    assert 'peft==0.20.0' in content
    assert 'numpy==2.1.3' in content
    assert 'do NOT pip install' in content


def test_no_notebook_performs_package_installations():
    root=Path(__file__).resolve().parents[1]
    notebooks=sorted((root/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==13
    for f in notebooks:
        data=json.loads(f.read_text())
        setup=''.join(data['cells'][1]['source'])
        assert 'EXPECTED' in setup and '5.16.1' in setup, f.name
        assert 'pip", "install' not in setup, f.name
        assert 'pip install' not in setup, f.name
        assert 'check_training_api()' in setup, f.name
