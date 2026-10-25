"""استعادة أرشيف المصنف الذي صدرناه، مع فحص المسارات وتوقيع البيانات."""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from data_utils import verify_classifier_info
from classification_config import MODELS, model_folder


def restore(archive, root, model='arabert'):
    root = Path(root)
    out = root/'outputs'
    out.mkdir(parents=True, exist_ok=True)
    target = model_folder(root, model)
    if list(target.glob('model*.safetensors')) or list(target.glob('pytorch_model*.bin')):
        raise FileExistsError('المصنف موجود بالفعل؛ لا نستبدل الأوزان بصمت.')
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for item in z.namelist():
                if not (tmp/item).resolve().is_relative_to(tmp.resolve()):
                    raise ValueError('مسار غير آمن داخل الأرشيف.')
            z.extractall(tmp)
        verify_classifier_info(tmp, root/'data')
        record = tmp/'training_info.json' if (tmp/'training_info.json').exists() else tmp/'experiment.json'
        info = json.loads(record.read_text(encoding='utf-8'))
        if info['model'] != MODELS[model]:
            raise ValueError('نوع النموذج في الأرشيف لا يطابق اسم المصنف المطلوب استعادته.')
        if not (tmp/'config.json').exists() or not (list(tmp.glob('model*.safetensors')) or list(tmp.glob('pytorch_model*.bin'))):
            raise ValueError('الأرشيف لا يحتوي أوزان المصنف وتكوينه في المستوى الأول.')
        shutil.copytree(tmp, target, dirs_exist_ok=True)
    print('Restored matching classifier:', target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--model', choices=list(MODELS), default='arabert')
    args = p.parse_args()
    restore(args.archive, args.root, args.model)
