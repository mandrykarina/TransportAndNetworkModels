"""LP потоков на пространственно-временной сети (склады × недели)."""
import numpy as np
from scipy.optimize import linprog

def build(data):
    weeks=data['weeks']; n=len(data['warehouses']);e=len(data['arcs'])
    # x[t,k] -- отправлено по дуге k в неделю t; inv[t,i] -- остаток в конце недели.
    count=weeks*(e+n)
    def ix(t,k):return t*e+k
    def ii(t,i):return weeks*e+t*n+i
    c=np.zeros(count);Aeq=[];beq=[];Aub=[];bub=[];ub_names=[]
    bounds=[(0,None)]*count
    for t in range(weeks):
        for k,edge in enumerate(data['arcs']):
            c[ix(t,k)]=edge['cost_rub_per_unit'];r=np.zeros(count);r[ix(t,k)]=1
            Aub.append(r);bub.append(edge['capacity_units_per_week']);ub_names.append(f'arc:{t+1}:{edge["from"]+1}>{edge["to"]+1}')
        for i in range(n):
            c[ii(t,i)]=data['holding_cost_rub_per_unit_week'][i]
            r=np.zeros(count);r[ii(t,i)]=1
            Aub.append(r);bub.append(data['storage_capacity_units'][i]);ub_names.append(f'storage:{t+1}:{i+1}')
            # Остаток(t) - остаток(t-1) + outgoing - incoming = inbound - demand.
            row=np.zeros(count);row[ii(t,i)]=1
            if t>0:row[ii(t-1,i)]=-1
            for k,edge in enumerate(data['arcs']):
                if edge['from']==i:row[ix(t,k)]+=1
                if edge['to']==i:row[ix(t,k)]-=1
            rhs=data['inbound_units_by_week'][t][i]-data['demand_units_by_week'][t][i]
            if t==0:rhs+=data['initial_inventory_units'][i]
            Aeq.append(row);beq.append(rhs)
    return {'c':c,'A_eq':np.asarray(Aeq),'b_eq':np.asarray(beq),
            'A_ub':np.asarray(Aub),'b_ub':np.asarray(bub),'bounds':bounds,
            'names':ub_names,'weeks':weeks,'n':n,'e':e}

def solve(m):
    return linprog(m['c'],A_eq=m['A_eq'],b_eq=m['b_eq'],
                   A_ub=m['A_ub'],b_ub=m['b_ub'],bounds=m['bounds'],method='highs-ds',
                   options={'presolve':True,'primal_feasibility_tolerance':1e-8,
                            'dual_feasibility_tolerance':1e-8})

def scenario(data,name,busiest=None):
    from copy import deepcopy
    d=deepcopy(data)
    if name=='base':pass
    elif name=='congestion':
        if busiest is None:raise ValueError('Не задана загруженная дуга')
        d['arcs'][busiest]['capacity_units_per_week']*=.5
    elif name=='closure':
        if busiest is None:raise ValueError('Не задана загруженная дуга')
        d['arcs'][busiest]['capacity_units_per_week']=0
    elif name=='demand_plus_20':
        d['demand_units_by_week']=(np.asarray(d['demand_units_by_week'])*1.2).tolist()
    elif name=='stress':
        d['demand_units_by_week']=(np.asarray(d['demand_units_by_week'])*1.2).tolist()
        if busiest is None:raise ValueError('Не задана загруженная дуга')
        d['arcs'][busiest]['capacity_units_per_week']=0
    else:raise ValueError(name)
    return d
