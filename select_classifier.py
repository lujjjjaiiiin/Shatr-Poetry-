"""نقارن النماذج المدربة على نفس أمثلة التحقق، ثم نختار أعلى Macro-F1."""
import json
import os
from pathlib import Path
import pandas as pd
from classification_config import MODELS, FOLDERS, model_folder
from data_utils import verify_training_info, data_signature


def select(root):
    root = Path(root)
    rows, reference = [], None
    for name in MODELS:
        folder = model_folder(root, name)
        verify_training_info(folder, root/'data')
        info = json.loads((folder/'training_info.json').read_text(encoding='utf-8'))
        if info.get('smoke') or info['model'] != MODELS[name]:
            raise ValueError('المقارنة تحتاج النموذجين الحقيقيين، لا فحص دخان أو نموذجًا آخر.')
        predictions = pd.read_csv(folder/'validation_predictions.csv')[['poem_id','text','label']]
        if reference is None:
            reference = predictions
        elif not reference.equals(predictions):
            raise ValueError('أمثلة التحقق أو ترتيبها مختلفة؛ لا نعتمد هذه المقارنة.')
        report = json.loads((folder/'classification_validation.json').read_text(encoding='utf-8'))
        rows.append(dict(model_key=name, model_id=MODELS[name], accuracy=report['accuracy'],
                         macro_f1=report['macro avg']['f1-score']))
    table = pd.DataFrame(rows).sort_values('macro_f1', ascending=False, kind='stable')
    best = table.iloc[0].to_dict()
    best.update(folder=FOLDERS[best['model_key']], criterion='validation_macro_f1',
                data_signature=data_signature(root/'data'))
    table.to_csv(root/'outputs/classifier_comparison.csv', index=False, encoding='utf-8-sig')
    (root/'outputs/selected_classifier.json').write_text(json.dumps(best,ensure_ascii=False,indent=2),encoding='utf-8')
    print(table.to_string(index=False))
    print('SELECTED_CLASSIFIER',json.dumps(best,ensure_ascii=False))
    return best


if __name__ == '__main__':
    select(Path(os.environ.get('SHATR_ROOT','.')))
