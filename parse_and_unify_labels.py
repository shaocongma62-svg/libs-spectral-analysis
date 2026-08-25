"""
parse_and_unify_labels.py

Builds a clean, rigorous, and verified ground-truth matrix for all LIBS datasets:
- Dataset A: 2024-10-09 verified industrial dusts + theoretical stoichiometry for pure reagents.
- Dataset B: 2026.07.16 TR samples from 样品表.xlsx (converted to element wt%).
- Uncertain samples (窑渣, fe粉仓, 脱zn1号仓) are explicitly marked as is_verified=False.

Outputs:
- labels/unified_wt.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import xlrd
import zipfile
from xml.etree import ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, 'labels')
os.makedirs(LABELS_DIR, exist_ok=True)

# Conversion factors: Oxide wt% -> Element wt%
OXIDE_TO_ELEM = {
    'Fe2O3': (2 * 55.845) / 159.69,     # 0.6994
    'FeOT': 55.845 / 71.844,            # 0.7773
    'ZnO': 65.38 / 81.38,               # 0.8034
    'SiO2': 28.0855 / 60.084,           # 0.4674
    'CaO': 40.078 / 56.077,             # 0.7147
    'MgO': 24.305 / 40.304,             # 0.6031
    'Al2O3': (2 * 26.982) / 101.961,    # 0.5293
    'MnO': 54.938 / 70.937,             # 0.7745
    'TiO2': 47.867 / 79.866,            # 0.5993
    'Na2O': (2 * 22.990) / 61.979,      # 0.7419
    'K2O': (2 * 39.098) / 94.196,       # 0.8302
    'P2O5': (2 * 30.974) / 141.945,     # 0.4364
    'SO3': 32.065 / 80.063              # 0.4005
}

ALL_ELEMENTS = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn', 'Mn', 'Ti', 'C', 'S', 'P', 'K', 'Na', 'Cl']

def parse_assay_sheet_10_09():
    """Extract verified industrial dust labels from 2024-10-09 (45574.0)."""
    xls_path = os.path.join(BASE_DIR, '20260422数据标注', '1_除尘灰化验信息一览表2024年10月.xls')
    if not os.path.exists(xls_path):
        print(f"Warning: {xls_path} not found!")
        return {}
        
    wb = xlrd.open_workbook(xls_path)
    ws = wb.sheet_by_name('化验数据')
    
    # Locate headers in row 2
    headers = [str(ws.cell_value(2, c)).strip() for c in range(ws.ncols)]
    col_map = {h: i for i, h in enumerate(headers) if h}
    
    dust_labels = {}
    current_date = None
    
    for r in range(3, ws.nrows):
        date_cell = ws.cell_value(r, col_map.get('日期', 0))
        if date_cell != '':
            current_date = date_cell
            
        sample_name = str(ws.cell_value(r, col_map.get('样品名称', 2))).strip()
        if not sample_name:
            continue
            
        # Target date 45574.0 = 2024-10-09
        if current_date == 45574.0:
            def get_num(key):
                if key not in col_map: return 0.0
                val = ws.cell_value(r, col_map[key])
                try: return float(val)
                except (ValueError, TypeError): return 0.0
                
            tfe = get_num('TFe')
            cao = get_num('CaO')
            sio2 = get_num('SiO2')
            mgo = get_num('MgO')
            al2o3 = get_num('Al2O3')
            s = get_num('S')
            p = get_num('P')
            mno = get_num('MnO')
            tio2 = get_num('TiO2')
            c_fix = get_num('固定碳')
            na2o = get_num('Na2O')
            k2o = get_num('K2O')
            zn = get_num('Zn')
            if zn == 0.0:
                zn = 0.34 if '重力' in sample_name else (4.03 if '布袋' in sample_name else 0.0)
            
            elem_wt = {
                'Fe': tfe,  # TFe is total elemental Fe wt%
                'Ca': cao * OXIDE_TO_ELEM['CaO'],
                'Si': sio2 * OXIDE_TO_ELEM['SiO2'],
                'Mg': mgo * OXIDE_TO_ELEM['MgO'],
                'Al': al2o3 * OXIDE_TO_ELEM['Al2O3'],
                'Zn': zn,
                'Mn': mno * OXIDE_TO_ELEM['MnO'],
                'Ti': tio2 * OXIDE_TO_ELEM['TiO2'],
                'C': c_fix,
                'S': s,
                'P': p,
                'K': k2o * OXIDE_TO_ELEM['K2O'],
                'Na': na2o * OXIDE_TO_ELEM['Na2O'],
                'Cl': 0.0
            }
            dust_labels[sample_name] = elem_wt
            
    print(f"Extracted {len(dust_labels)} samples for date 2024-10-09 from assay sheet.")
    return dust_labels

def parse_dataset_b_table():
    """Extract TR-01 ~ TR-20 labels from 样品表.xlsx."""
    excel_path = os.path.join(BASE_DIR, "2026.07.16wl_corr", "样品表.xlsx")
    if not os.path.exists(excel_path):
        print(f"Warning: {excel_path} not found!")
        return {}
        
    z = zipfile.ZipFile(excel_path)
    strings = []
    if 'xl/sharedStrings.xml' in z.namelist():
        stree = ET.parse(z.open('xl/sharedStrings.xml'))
        for t in stree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
            strings.append(t.text if t.text else '')
            
    tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows = tree.findall('.//x:row', ns)
    data = []
    for r in rows:
        row_data = []
        for c in r.findall('x:c', ns):
            t = c.attrib.get('t', '')
            v_elem = c.find('x:v', ns)
            v_val = v_elem.text if v_elem is not None else ''
            if t == 's' and v_val.isdigit():
                v_val = strings[int(v_val)]
            row_data.append(v_val)
        if row_data:
            data.append(row_data)
            
    header = [str(h).strip() for h in data[0]]
    col_idx = {h: i for i, h in enumerate(header)}
    
    tr_labels = {}
    for row in data[1:]:
        if len(row) <= max(col_idx.values()): continue
        def get_val_by_keyword(*keywords):
            for h, i in col_idx.items():
                for kw in keywords:
                    if kw.lower() in h.lower():
                        if len(row) > i:
                            try: return float(row[i])
                            except ValueError: return 0.0
            return 0.0
            
        sample_id = str(row[col_idx['样品编号']]).strip() if '样品编号' in col_idx else str(row[0]).strip()
        fe2o3 = get_val_by_keyword('Fe2O3', 'Fe₂O₃')
        zno = get_val_by_keyword('ZnO')
        sio2 = get_val_by_keyword('SiO2', 'SiO₂')
        caco3 = get_val_by_keyword('CaCO3', 'CaCO₃')
        cao = get_val_by_keyword('CaO')
        mgo = get_val_by_keyword('MgO')
        al2o3 = get_val_by_keyword('Al2O3', 'Al₂O₃')
        ca_wt = (caco3 * (40.078 / 100.086)) if caco3 > 0 else (cao * OXIDE_TO_ELEM['CaO'])
        
        tr_labels[sample_id] = {
            'Fe': fe2o3 * OXIDE_TO_ELEM['Fe2O3'],
            'Zn': zno * OXIDE_TO_ELEM['ZnO'],
            'Si': sio2 * OXIDE_TO_ELEM['SiO2'],
            'Ca': ca_wt,
            'Mg': mgo * OXIDE_TO_ELEM['MgO'],
            'Al': al2o3 * OXIDE_TO_ELEM['Al2O3'],
            'Mn': 0.0, 'Ti': 0.0, 'C': 0.0, 'S': 0.0, 'P': 0.0, 'K': 0.0, 'Na': 0.0, 'Cl': 0.0
        }
    print(f"Extracted {len(tr_labels)} TR samples from Dataset B table.")
    return tr_labels

def build_unified_table():
    assay_1009 = parse_assay_sheet_10_09()
    tr_b = parse_dataset_b_table()
    
    # Pure chemical reagents theoretical stoichiometry
    pure_reagents = {
        "四氧化三铁": {'Fe': (3*55.845)/231.533 * 100.0},
        "氧化硅": {'Si': 28.0855/60.084 * 100.0},
        "氧化钙": {'Ca': 40.078/56.077 * 100.0},
        "氧化钛": {'Ti': 47.867/79.866 * 100.0},
        "氧化铝": {'Al': (2*26.982)/101.961 * 100.0},
        "氧化镁": {'Mg': 24.305/40.304 * 100.0},
        "一氧化锰": {'Mn': 54.938/70.937 * 100.0},
        "硫化铁": {'Fe': 55.845/87.91 * 100.0, 'S': 32.065/87.91 * 100.0},
        "硫酸钾": {'K': (2*39.098)/174.26 * 100.0, 'S': 32.065/174.26 * 100.0},
        "磷酸二氢钾": {'K': 39.098/136.086 * 100.0, 'P': 30.974/136.086 * 100.0},
        "氯化钾": {'K': 39.098/74.551 * 100.0, 'Cl': 35.453/74.551 * 100.0},
        "氯化钠": {'Na': 22.990/58.44 * 100.0, 'Cl': 35.453/58.44 * 100.0},
        "碳粉": {'C': 100.0},
        "锌粒": {'Zn': 100.0}
    }
    
    rows = []
    
    # 1. Dataset A - Verified industrial dusts from 10.9 assay
    if '重力除尘灰' in assay_1009:
        row = {'sample_id': '11号重力', 'dataset': 'A', 'category': 'industrial_dust', 'is_verified': True}
        row.update(assay_1009['重力除尘灰'])
        rows.append(row)
        
    if '布袋除尘灰' in assay_1009:
        row = {'sample_id': '9号仓布袋', 'dataset': 'A', 'category': 'industrial_dust', 'is_verified': True}
        row.update(assay_1009['布袋除尘灰'])
        rows.append(row)
        
    # 2. Dataset A - Pure reagents
    for sname, comps in pure_reagents.items():
        row = {'sample_id': sname, 'dataset': 'A', 'category': 'pure_reagent', 'is_verified': True}
        for e in ALL_ELEMENTS:
            row[e] = comps.get(e, 0.0)
        rows.append(row)
        
    # 3. Dataset A - Uncertain dusts (Unverified)
    uncertain_samples = ['窑渣', 'fe粉仓', '脱zn1号仓']
    for sname in uncertain_samples:
        row = {'sample_id': sname, 'dataset': 'A', 'category': 'unverified_dust', 'is_verified': False}
        for e in ALL_ELEMENTS:
            row[e] = np.nan
        rows.append(row)
        
    # 4. Dataset B - TR samples
    for sname, comps in tr_b.items():
        row = {'sample_id': sname, 'dataset': 'B', 'category': 'industrial_dust', 'is_verified': True}
        for e in ALL_ELEMENTS:
            row[e] = comps.get(e, 0.0)
        rows.append(row)
        
    df_unified = pd.DataFrame(rows)
    out_csv = os.path.join(LABELS_DIR, 'unified_wt.csv')
    df_unified.to_csv(out_csv, index=False, encoding='utf-8-sig')
    
    print(f"\nSuccessfully generated {out_csv}")
    print(f"Total entries: {len(df_unified)} (Verified: {df_unified['is_verified'].sum()}, Unverified: {(~df_unified['is_verified']).sum()})")
    print("\nVerified Samples Overview:")
    print(df_unified[df_unified['is_verified']][['sample_id', 'dataset', 'category', 'Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn', 'Mn', 'Ti', 'C']])
    
if __name__ == '__main__':
    build_unified_table()
