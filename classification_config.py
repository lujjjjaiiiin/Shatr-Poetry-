"""اسمان واضحان للنموذجين؛ لكل نموذج مجلد مستقل حتى لا نستبدل أوزانه."""
import json
from pathlib import Path

MODELS = {
    'arabert': 'aubmindlab/bert-base-arabertv02',
    'arapoembert': 'faisalq/bert-base-arapoembert',
}
FOLDERS = {'arabert': 'basma_model', 'arapoembert': 'arapoembert_model'}


def model_folder(root, name, smoke=False):
    folder = FOLDERS[name]
    if smoke:
        folder = 'basma_smoke' if name == 'arabert' else 'arapoembert_smoke'
    return Path(root)/'outputs'/folder


def selected_folder(root):
    root = Path(root)
    selection = root/'outputs/selected_classifier.json'
    name = json.loads(selection.read_text(encoding='utf-8'))['model_key'] if selection.exists() else 'arabert'
    if name not in MODELS:
        raise ValueError('اسم المصنف المختار غير معروف.')
    return model_folder(root, name)
