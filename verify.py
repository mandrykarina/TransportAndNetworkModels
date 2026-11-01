"""Полная повторная проверка материального баланса, ёмкостей и стоимости по JSON."""
import numpy as np

def check(data,flows,inventories,tol=1e-7):
    f=np.asarray(flows,dtype=float);v=np.asarray(inventories,dtype=float)
    T=data['weeks'];n=len(data['warehouses']);E=len(data['arcs'])
    assert f.shape==(T,E) and v.shape==(T,n)
    capacities=np.array([e['capacity_units_per_week'] for e in data['arcs']])
    residual=[]
    for t in range(T):
        for i in range(n):
            incoming=sum(f[t,k] for k,e in enumerate(data['arcs']) if e['to']==i)
            outgoing=sum(f[t,k] for k,e in enumerate(data['arcs']) if e['from']==i)
            prev=data['initial_inventory_units'][i] if t==0 else v[t-1,i]
            residual.append(v[t,i]-(prev+data['inbound_units_by_week'][t][i]+incoming-outgoing-data['demand_units_by_week'][t][i]))
    violations={
       'negative_flow':float(np.maximum(0,-f).max(initial=0)),
       'negative_stock':float(np.maximum(0,-v).max(initial=0)),
       'arc_capacity':float(np.maximum(0,f-capacities).max(initial=0)),
       'storage_capacity':float(np.maximum(0,v-np.array(data['storage_capacity_units'])).max(initial=0)),
       'node_balance':float(np.max(np.abs(residual),initial=0))}
    costs=np.array([e['cost_rub_per_unit'] for e in data['arcs']]);hold=np.asarray(data['holding_cost_rub_per_unit_week'])
    return {'valid':all(z<=tol for z in violations.values()),'violations':violations,
            'max_violation':max(violations.values()),
            'objective_rub':float(np.sum(f*costs)+np.sum(v*hold)),
            'final_inventory_units':v[-1].tolist()}
