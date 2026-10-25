"""فحوص سلوك مهمة؛ لا تحمل نماذج أو تصنع نتائج للمشروع."""
import json
import tempfile
import zipfile
import io
import hashlib
from contextlib import redirect_stdout
from pathlib import Path
import pandas as pd
from app import ShatrService, build_app, cards, safe_source
from data_utils import guard_run, write_training_info, verify_classifier_info
from evaluate_humans import summarize, CRITERIA
from restore_model import restore
from classification_config import MODELS, model_folder, selected_folder
from select_classifier import select


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('كان يجب رفض البيانات غير الصالحة.')


def main():
    assert safe_source('javascript:alert(1)') == ''
    assert safe_source('https://example.org/poem') == 'https://example.org/poem'
    rendered = cards([dict(text='<script>alert(1)</script>', poet='<b>شاعر</b>', era='عصر', source='javascript:alert(1)')])
    assert '<script>' not in rendered and 'javascript:' not in rendered and '&lt;script&gt;' in rendered
    key = pd.DataFrame([dict(id=1, version='A', actual_version='after'), dict(id=1, version='B', actual_version='before')])
    ratings = pd.DataFrame([dict(id=1, version=v, **{c:score for c in CRITERIA}) for v,score in [('A',4),('B',2)]])
    result, _ = summarize(ratings, key)
    assert result.loc['after','meaning'] == 4 and result.loc['before','meaning'] == 2
    broken = ratings.copy(); broken.loc[0,'meaning'] = None
    rejects(lambda:summarize(broken,key))
    rejects(lambda:summarize(ratings.iloc[:1],key))
    rejects(lambda:summarize(pd.concat([ratings,ratings]),key))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        service = ShatrService(root)
        assert not service.classifier_ready and not service.completion_ready and service.corpus is None
        # الواجهة تُبنى حتى عندما لم يُدرب أي نموذج بعد.
        demo = build_app(root)
        assert demo is not None
        data = root/'data'; data.mkdir()
        rows = [
            dict(text='يا ليل قلبي في هواك مسافر',poet='شاعر أول',era='عصر أ',source='https://example.org/1'),
            dict(text='يا ليل قلبي في هواك حائر',poet='شاعر أول',era='عصر أ',source='https://example.org/2'),
            dict(text='يا ليل قلبي في الحنين قصيدة',poet='شاعر ثان',era='عصر أ',source='https://example.org/3'),
        ]
        frame = pd.DataFrame(rows)
        for name in ['labels.json'] + [f'{task}_{split}.jsonl' for task in ['cls','gen'] for split in ['train','validation','test']]:
            (data/name).write_text('{}\n', encoding='utf-8')
        frame.to_json(data/'cls_train.jsonl',orient='records',lines=True,force_ascii=False)
        service = ShatrService(root)
        result = service.search('يا ليل قلبي في هواك مسافر')
        assert 'شاعر أول' in result and 'شاعر ثان' in result
        assert result.count('class="result-card"') == 2
        assert 'في هواك مسافر' not in result  # نستبعد التطابق التام.
        legacy = root/'legacy_model'; legacy.mkdir()
        hashes = {n:hashlib.sha256((data/n).read_bytes()).hexdigest()
                  for n in ['cls_train.jsonl','cls_validation.jsonl','labels.json']}
        (legacy/'experiment.json').write_text(json.dumps({'data_sha256':hashes}),encoding='utf-8')
        verify_classifier_info(legacy,data)
        original = (data/'labels.json').read_bytes()
        (data/'labels.json').write_text('{"changed":0}',encoding='utf-8')
        rejects(lambda:verify_classifier_info(legacy,data))
        (data/'labels.json').write_bytes(original)
        run = root/'run'
        guard_run(run,data,'example',{'seed':42})
        guard_run(run,data,'example',{'seed':42})
        rejects(lambda:guard_run(run,data,'example',{'seed':43}))
        assert model_folder(root,'arabert') != model_folder(root,'arapoembert')
        predictions = pd.DataFrame([dict(poem_id='p1',text='بيت اختباري',label=0)])
        for i, name in enumerate(MODELS):
            folder = model_folder(root,name)
            write_training_info(folder,data,MODELS[name])
            predictions.to_csv(folder/'validation_predictions.csv',index=False)
            (folder/'classification_validation.json').write_text(json.dumps(
                {'accuracy':.5,'macro avg':{'f1-score':.4+i*.1}}),encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            chosen = select(root)
        assert chosen['model_key'] == 'arapoembert'
        assert selected_folder(root) == model_folder(root,'arapoembert')
        different = predictions.copy(); different.loc[0,'text'] = 'بيت مختلف'
        different.to_csv(model_folder(root,'arapoembert')/'validation_predictions.csv',index=False)
        rejects(lambda:select(root))
        # نرفض traversal قبل فك أي عضو في أرشيف غير آمن.
        archive = root/'bad.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('../escaped.txt','unsafe')
        rejects(lambda:restore(archive,root))
        assert not (root/'escaped.txt').exists()
    print('PASS: safe source links, escaped results, diverse retrieval, exact-match exclusion,')
    print('      incomplete-model UI, strict human ratings and resume guards.')


if __name__ == '__main__':
    main()
