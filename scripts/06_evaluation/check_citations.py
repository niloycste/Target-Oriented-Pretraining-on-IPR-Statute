import re
from pathlib import Path

CITE_PAT = re.compile(r'[\\]cite[tp]?[{]([^}]+)[}]')
BIB_PAT  = re.compile(r'@\w+[{]([^,\n]+),')
BBI_PAT  = re.compile(r'[\\]bibitem[{]([^}]+)[}]')

def get_cites(path):
    txt = open(path, encoding='utf-8').read()
    # remove comments
    txt = re.sub(r'%[^\n]*', '', txt)
    keys = set()
    for m in CITE_PAT.finditer(txt):
        for k in m.group(1).split(','):
            key = k.strip()
            if key and not key.startswith('#'):
                keys.add(key)
    return keys

def get_bib(path):
    txt = open(path, encoding='utf-8').read()
    return set(m.group(1).strip() for m in BIB_PAT.finditer(txt))

def get_bibitems(path):
    txt = open(path, encoding='utf-8').read()
    return set(m.group(1).strip() for m in BBI_PAT.finditer(txt))

ROOT = Path(__file__).resolve().parents[2]
papers = [
    (
        'CURRENT SPRINGER PAPER',
        ROOT / 'combined_paper' / 'Springer_Nature_LaTeX_Template' / 'sn-article.tex',
        ROOT / 'combined_paper' / 'Springer_Nature_LaTeX_Template' / 'references.bib',
        False,
    ),
    (
        'LEGACY COMBINED PAPER',
        ROOT / 'combined_paper' / 'main.tex',
        ROOT / 'combined_paper' / 'references.bib',
        False,
    ),
]

for title, tex, bib, inline in papers:
    print('=' * 60)
    print(title)
    print('=' * 60)
    cites = get_cites(tex)
    bibs  = get_bibitems(tex) if inline else get_bib(bib)
    missing = sorted(cites - bibs)
    unused  = sorted(bibs  - cites)
    print(f'  Cited   : {len(cites)}')
    print(f'  Defined : {len(bibs)}')
    if missing:
        print(f'  MISSING (cited but no bib entry) : {missing}')
    else:
        print(f'  MISSING : None')
    if unused:
        print(f'  Unused  (defined but never cited): {unused}')
    else:
        print(f'  Unused  : None')
    print()
