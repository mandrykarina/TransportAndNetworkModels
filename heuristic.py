"""Жадный алгоритм перераспределения: ближайший дешёвый доступный путь."""
import time
import numpy as np
import networkx as nx
from verify import check

def greedy(data):
    start=time.perf_counter()
    n=len(data['warehouses']);T=data['weeks'];E=len(data['arcs'])
    shipments=np.zeros((T,E));stocks=np.zeros((T,n))
    prev=np.asarray(data['initial_inventory_units'],float)
    for t in range(T):
        balance=prev+np.asarray(data['inbound_units_by_week'][t])-np.asarray(data['demand_units_by_week'][t])
        remaining=np.array([a['capacity_units_per_week'] for a in data['arcs']],float)
        for _ in range(1000):
            shortage=np.where(balance< -1e-9)[0]
            if len(shortage)==0:break
            donors=np.where(balance>1e-9)[0]
            best=None
            G=nx.DiGraph()
            for k,a in enumerate(data['arcs']):
                if remaining[k]>1e-9:G.add_edge(a['from'],a['to'],weight=a['cost_rub_per_unit'],idx=k)
            for i in donors:
                lengths,paths=nx.single_source_dijkstra(G,int(i),weight='weight') if i in G else ({},{})
                for j in shortage:
                    if int(j) in lengths:
                        item=(lengths[int(j)],int(i),int(j),paths[int(j)])
                        if best is None or item[:3]<best[:3]:best=item
            if best is None:
                return {'valid':False,'reason':f'На неделе {t+1} жадный алгоритм не нашёл маршрута','seconds':time.perf_counter()-start}
            _,i,j,path=best
            edge_ids=[G[path[h]][path[h+1]]['idx'] for h in range(len(path)-1)]
            amount=min(balance[i],-balance[j],*(remaining[k] for k in edge_ids))
            if amount<=1e-10:raise AssertionError('Жадный поток не продвинулся')
            for k in edge_ids:shipments[t,k]+=amount;remaining[k]-=amount
            balance[i]-=amount;balance[j]+=amount
        if np.any(balance< -1e-7):
            return {'valid':False,'reason':'Часть спроса не закрыта','seconds':time.perf_counter()-start}
        if np.any(balance>np.asarray(data['storage_capacity_units'])+1e-7):
            return {'valid':False,'reason':'Переполнение складов','seconds':time.perf_counter()-start}
        stocks[t]=balance;prev=balance.copy()
    v=check(data,shipments,stocks)
    return {'valid':v['valid'],'objective_rub':v['objective_rub'],'seconds':time.perf_counter()-start,
            'flows':shipments.tolist(),'inventories':stocks.tolist()}
