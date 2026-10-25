"""Generate a report from actual result files; missing results stay explicitly missing."""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from classification_config import selected_folder

ROOT = Path(os.environ.get('SHATR_ROOT', '.'))

def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def main():
    out = ROOT/'outputs'
    report_dir = ROOT/'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline = load(out/'baseline_validation.json')
    arabert = load(out/'basma_model/classification_validation.json')
    arapoembert = load(out/'arapoembert_model/classification_validation.json')
    selected = selected_folder(ROOT)
    model = load(selected/'classification_validation.json')
    text = ['# شطر — تقرير النتائج', '',
        '## المشكلة والهدف',
        'مساعدة القارئ على استكشاف العصر الأقرب لنص شعري، ومساعدة الكاتب على إكمال شطره الأول.', '',
        '## المنهج',
        'البيانات من Ashaar. استُخدمت الأزواج التي جرى تجهيزها ومراجعتها في الدفتر الأول. '
        'أُعيد التقسيم مرة واحدة لمعالجة ضعف تمثيل الشعراء، دون اختيار البذرة بناءً على درجات النماذج. '
        'التقسيم الجديد يفصل الشعراء داخل كل عصر. تُزال تكرارات الأشطر بين الأقسام، '
        'وتؤخذ عينات تراعي تنوع الشعراء. تغيّر مجموعة التحقق يعني أن المقارنة بالتجربة القديمة ليست مقارنة ثابتة البيانات.', '',
        '## نتائج التحقق', '',
        '| النموذج | Accuracy | Macro-F1 |', '|---|---:|---:|']
    for name, result in [('TF-IDF + Logistic Regression', baseline), ('AraBERT', arabert), ('AraPoemBERT', arapoembert)]:
        if result:
            text.append(f"| {name} | {result['accuracy']:.2%} | {result['macro avg']['f1-score']:.2%} |")
        else:
            text.append(f'| {name} | لم ينفذ بعد | لم ينفذ بعد |')
    labels = load(ROOT/'data/labels.json')
    if labels:
        for name, result in [('AraBERT', arabert), ('AraPoemBERT', arapoembert)]:
            if result:
                text += ['', f'### نتائج كل عصر — {name}', '',
                         '| العصر | Precision | Recall | F1 |', '|---|---:|---:|---:|']
                for era in labels:
                    row = result[era]
                    text.append(f"| {era} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1-score']:.2%} |")
                pd.DataFrame(result['confusion_matrix'],index=list(labels),columns=list(labels)).to_csv(
                    report_dir/f'confusion_matrix_{name}.csv',encoding='utf-8-sig')
    selection = load(out/'selected_classifier.json')
    if selection and selection.get('data_profile') == 'legacy':
        text[6] = ('النسخة الحالية تستخدم التقسيم الأصلي دون إعادة تقسيم أو تدريب المصنف. '
                   'نتيجة AraPoemBERT محفوظة من التجربة الأصلية؛ تحقق التواقيع الأصلية يربطها '
                   'بملفات التدريب والتحقق والتسميات. لا تنسب إليها شروط تمثيل التقسيم الجديد.')
    if selection:
        text += ['', 'المصنف المعتمد للواجهة: **' + selection['model_id'] + '** وفق Macro-F1 على أمثلة التحقق نفسها.']
    if model and labels:
        names = list(labels)
        matrix = np.array(model['confusion_matrix'])
        pd.DataFrame(matrix,index=names,columns=names).to_csv(report_dir/'confusion_matrix.csv',encoding='utf-8-sig')
        # Numeric IDs avoid Arabic shaping differences in chart environments.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,5))
        ax.imshow(matrix,cmap='YlOrBr')
        ax.set(xticks=range(len(names)),yticks=range(len(names)),
               xlabel='Predicted era ID',ylabel='True era ID',title='Validation confusion matrix')
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',
                        color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        fig.tight_layout()
        fig.savefig(report_dir/'confusion_matrix.png',dpi=160)
        plt.close(fig)
        text += ['', 'ترتيب معرّفات العصور في الرسم:']
        text += [f'- {i}: {name}' for i,name in enumerate(names)]
        text += ['', '![Validation confusion matrix](confusion_matrix.png)']
    if (ROOT/'data/data_summary.csv').exists():
        text += ['', '## توزيع البيانات', '', '```text',
                 pd.read_csv(ROOT/'data/data_summary.csv').to_string(index=False), '```']
    text += ['', '## تقييم شطر',
        'ملف shatr_compare.csv يعرض التكملات قبل التدريب وبعده. '
        'ملف human_eval_validation.csv مخصص لتقييم أعمى من ١ إلى ٥ للغة والمعنى والطابع الشعري وغياب التكرار. '
        'ملف human_eval_key_validation.csv يكشف ترتيب النسخ ويحتفظ به المنسق بعيدًا عن المقيم. '
        'لا تُعد خسارة النموذج دليلًا على صحة الوزن أو جودة الشعر.', '',
        '## الاختبار النهائي',
        'شغّلوا final test بعد تثبيت القرارات. الأرقام أعلاه تخص validation فقط.']
    text += ['', 'حالة مقارنة التوليد: ' + ('الملف موجود؛ يلزم إتمام التقييم البشري.' if (out/'shatr_compare.csv').exists() else 'لم تُنفذ مقارنة التوليد بعد.') ]
    text += ['حالة التقييم البشري: ' + ('توجد درجات ملخصة؛ راجعي عدد المقيمين وأكتمال التقييم.' if (ROOT/'human_report/human_summary.csv').exists() else 'لا توجد درجات بشرية ملخصة لهذه النسخة.') ]
    final_model = load(out/'basma_test.json')
    if final_model:
        text += [f"نتيجة AraBERT النهائية: Accuracy={final_model['accuracy']:.2%}، "
                 f"Macro-F1={final_model['macro avg']['f1-score']:.2%}."]
    else:
        text += ['لم يُنفذ الاختبار النهائي بعد.']
    final_poem = load(out/'arapoembert_test.json')
    if final_poem:
        text += [f"نتيجة AraPoemBERT النهائية: Accuracy={final_poem['accuracy']:.2%}، "
                 f"Macro-F1={final_poem['macro avg']['f1-score']:.2%}."]
    text += ['', '## القيود',
        'التصنيف بين أربع فئات لا يثبت تاريخ النص. قد تؤثر قلة الشعراء وتداخل الأغراض الأدبية '
        'وقصر النص وجودة التسميات في الأداء. النموذج الأساسي قد يعرف قصائد مشهورة من تدريبه المسبق. '
        'النسخة الحالية لا تتحقق آليًا من العروض أو القافية، ولا تضمن أصالة التكملات.', '',
        '## المصادر',
        '- https://huggingface.co/datasets/arbml/ashaar',
        '- https://huggingface.co/aubmindlab/bert-base-arabertv02',
        '- https://huggingface.co/Qwen/Qwen3-0.6B',
        '- https://huggingface.co/faisalq/bert-base-arapoembert']
    (report_dir/'Project_Report.md').write_text('\n'.join(text),encoding='utf-8')
    print('Report:',report_dir/'Project_Report.md')

if __name__ == '__main__':
    main()
