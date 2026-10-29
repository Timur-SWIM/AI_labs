#!/usr/bin/env python3
"""Automated journal-fill pipeline: analyse → collect → preview → final → validate."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'journal_fill/scripts'
PYTHON=sys.executable

def run(name, *args): subprocess.run([PYTHON,str(SCRIPTS/name),*args],check=True,cwd=ROOT)

def main():
    for name in ('inspect_pdf.py','find_lab_pages.py','collect_lab_results.py','build_field_specs.py'): run(name)
    data=json.loads((ROOT/'journal_fill/collected_results.json').read_text(encoding='utf-8'))
    lines=['# Инвентаризация лабораторных работ','']
    for lab,info in data.items():
        lines += [f'## {lab.upper()}',f'- каталог: {info.get("directory") or "не найден"}',f'- статус: {info["status"]}',
                  f'- run_all.py: {"да" if info.get("run_all") else "нет"}',f'- report/theory/answers: {info.get("report",False)}/{info.get("theory",False)}/{info.get("answers",False)}',
                  f'- CSV/JSON: {len(info.get("csv_json",[]))}; PNG: {len(info.get("png",[]))}',f'- причина: {info.get("reason", "—")}','']
    (ROOT/'journal_fill/lab_inventory.md').write_text('\n'.join(lines),encoding='utf-8')
    for lab in ('lab04','lab05','lab06'):
        fields=json.loads((ROOT/f'journal_fill/specs/{lab}.json').read_text(encoding='utf-8'))['items']
        details=[f'# LAB_SOURCE_SUMMARY — {lab.upper()}','']
        for item in fields: details += [f'Поле журнала: {item["name"]}',f'Источник: {item["source"]}',f'Страница PDF: {item["page"]}',f'Координаты: {item["rect"]}',f'Значение: {item.get("value","график из источника")}', '']
        (ROOT/f'journal_fill/lab_sources/{lab}.md').write_text('\n'.join(details),encoding='utf-8')
    unfilled=['# Незаполненные поля','', 'ЛР №7: все экспериментальные поля. Причина: каталог, вариантные данные и фактические результаты отсутствуют.',
              'ЛР №8: все экспериментальные поля. Причина: каталог и фактические результаты отсутствуют.',
              'ЛР №9: все экспериментальные поля. Причина: каталог и фактические результаты отсутствуют.',
              'Поля даты, оценки, бонуса и подписи ЛР №4–6 не заполнялись: персональные/оценочные данные не предоставлены.']
    (ROOT/'journal_fill/unfilled_fields.md').write_text('\n'.join(unfilled),encoding='utf-8')
    run('create_preview.py'); run('render_and_check.py'); run('fill_pdf.py'); run('validate_final.py')
    manifest=[]
    for spec in sorted((ROOT/'journal_fill/specs').glob('lab*.json')):
        for item in json.loads(spec.read_text(encoding='utf-8'))['items']:
            manifest.append({**item,'status':'filled','font':'DejaVu Sans' if item['kind']=='text' else None})
    (ROOT/'journal_fill/fill_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    logs=['# Журнал заполнения','', '## ЛР №4','Страницы: 25–31. Заполнены теория, параметры/метрики XOR, аппроксимация, классификация/Iris; вставлены реальные PNG Python-экспериментов.',
          '## ЛР №5','Страницы: 32–38. Заполнены теория Хемминга/MAXNET, сведения о трёх классах, ε и устойчивости; вставлены реальные PNG.',
          '## ЛР №6','Страницы: 39–43. Заполнены теория MLP/backpropagation, фактические MSE/GF/BF и выбранная архитектура; вставлены реальные PNG.',
          '## ЛР №7–9','Экспериментальные результаты не вставлялись: соответствующие лабораторные отсутствуют. См. unfilled_fields.md.',
          '', 'PDF overlay выполнен через qpdf + прозрачные растровые слои, поскольку PyMuPDF/pypdf/pikepdf/reportlab отсутствуют в окружении. Исходный PDF не изменялся.']
    (ROOT/'journal_fill/fill_log.md').write_text('\n'.join(logs),encoding='utf-8')
    print('PDF:',ROOT/'211000_iu4_sem09_sii_lab_rab_filled.pdf')
    print('LAB04/05/06: READY; LAB07/08/09: BLOCKED (no factual lab artifacts)')
    print('Validation: PASS')
if __name__=='__main__': main()

