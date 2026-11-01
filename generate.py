"""Генерация четырёхнедельной сети запасов с 8 складами."""
import json
from pathlib import Path
import numpy as np
SEED=10017

def make_data():
    rng=np.random.default_rng(SEED)
    nodes=[f'W{i+1}' for i in range(8)]
    routes=[]
    for i in range(8):
        for j in [(i+1)%8,(i-1)%8]:
            routes.append((i,j))
    routes+= [(0,3),(0,4),(1,4),(1,5),(2,5),(2,6),(2,7),(1,7),(0,6),(3,6)]
    routes=list(dict.fromkeys(routes))
    edges=[]
    for i,j in routes:
        edges.append({'from':i,'to':j,'capacity_units_per_week':int(rng.integers(18,33)),
                      'cost_rub_per_unit':float(np.round(1.3+min(abs(i-j),8-abs(i-j))*.35+rng.uniform(0,.7),3))})
    for a in edges:
        if (a['from'],a['to'])==(1,4):a['capacity_units_per_week']=17
        if (a['from'],a['to'])==(2,3):a['capacity_units_per_week']=18
    initial=[int(rng.integers(8,16)) for _ in range(8)]
    receipt=[];demand=[]
    for t in range(4):
        receipt.append([int(rng.integers(56,66)) if i<3 else int(rng.integers(4,9)) for i in range(8)])
        demand.append([int(rng.integers(18,24)) for _ in range(8)])
    return {'N':17,'Q':0,'Y':26,'variant':16,'seed':SEED,'weeks':4,'warehouses':nodes,
         'arcs':edges,'initial_inventory_units':initial,'inbound_units_by_week':receipt,
         'demand_units_by_week':demand,'storage_capacity_units':[250]*8,
         'holding_cost_rub_per_unit_week':[float(np.round(.15+.03*i,3)) for i in range(8)],
         'description':'Условные единицы однотипного товара; поставки и спрос заданы на каждую неделю.'}
if __name__=='__main__':
    f=Path(__file__).resolve().parent/'data'/'instance.json';f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(make_data(),ensure_ascii=False,indent=2),encoding='utf-8');print(f)
