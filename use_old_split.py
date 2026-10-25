"""تفعيل نموذج 61.20% مع بياناته الأصلية، دون إعادة تقسيم أو تدريب."""
import argparse
import json
import shutil
from pathlib import Path
from data_utils import verify_classifier_info


def activate(old_root, model, target):
    old_root, model, target = Path(old_root), Path(model), Path(target)
    verify_classifier_info(model, old_root/'data')
    metrics = json.loads((model/'validation_metrics.json').read_text(encoding='utf-8'))
    record = json.loads((model/'experiment.json').read_text(encoding='utf-8'))
    if record['model'] != 'faisalq/bert-base-arapoembert':
        raise ValueError('هذه الاستعادة مخصصة لتجربة AraPoemBERT الأصلية.')
    if (target/'data').exists():
        verify_classifier_info(model, target/'data')
    else:
        shutil.copytree(old_root/'data',target/'data')
    folder = target/'outputs/arapoembert_model'
    if folder.exists():
        verify_classifier_info(folder,target/'data')
    else:
        shutil.copytree(model,folder,ignore=shutil.ignore_patterns('checkpoints'))
    selection = dict(model_key='arapoembert', model_id=record['model'],folder='arapoembert_model',
        accuracy=metrics['eval_accuracy'],macro_f1=metrics['eval_macro_f1'],
        criterion='user selected original experiment',data_profile='legacy',
        data_sha256=record['data_sha256'],note='Original train/validation/labels hashes verified. No new training or split.')
    (target/'outputs/selected_classifier.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print('OLD_SPLIT_ACTIVE',json.dumps(selection,ensure_ascii=False))
    return selection


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--old-root',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--target',required=True)
    a=p.parse_args()
    activate(a.old_root,a.model,a.target)
