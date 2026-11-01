"""Решение складской сети, численные проверки, визуализация и эксперименты."""
import csv,json,platform
from pathlib import Path
import scipy,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from model import build,solve,scenario
from verify import check
from heuristic import greedy
ROOT=Path(__file__).resolve().parent

def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')

def main():
    base=json.loads((ROOT/'data'/'instance.json').read_text(encoding='utf8'))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    base_m=build(base);base_res=solve(base_m)
    if not base_res.success:raise RuntimeError('Базовая постановка несовместна: '+base_res.message)
    T,n,E=base_m['weeks'],base_m['n'],base_m['e']
    flow0=base_res.x[:T*E].reshape(T,E)
    most=int(np.argmax(flow0.sum(axis=0)))
    runs={};log=[]
    for name in ['base','congestion','closure','demand_plus_20','stress']:
        d=scenario(base,name,most);m=build(d);res=solve(m)
        row={'scenario':name,'status':{0:'optimal',1:'time_limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(res.status,'unknown'),
             'message':res.message,'iterations':int(getattr(res,'nit',0)),'objective_rub':None}
        if res.success:
            flow=res.x[:T*E].reshape(T,E);stocks=res.x[T*E:].reshape(T,n)
            verification=check(d,flow,stocks)
            if not verification['valid'] or abs(verification['objective_rub']-res.fun)>1e-6:raise RuntimeError(f'Неверное решение {name}: {verification}')
            row.update(objective_rub=float(res.fun),validation=verification)
            save(out/f'validation_{name}.json',verification)
            with (out/f'flows_{name}.csv').open('w',newline='',encoding='utf-8') as fp:
                w=csv.writer(fp);w.writerow(['week','from','to','flow_units','capacity_units','cost_rub_per_unit'])
                for t in range(T):
                    for k,a in enumerate(d['arcs']):w.writerow([t+1,d['warehouses'][a['from']],d['warehouses'][a['to']],f'{flow[t,k]:.9f}',a['capacity_units_per_week'],a['cost_rub_per_unit']])
            with (out/f'inventory_{name}.csv').open('w',newline='',encoding='utf-8') as fp:
                w=csv.writer(fp);w.writerow(['week','warehouse','inventory_end_units'])
                for t in range(T):
                    for i in range(n):w.writerow([t+1,d['warehouses'][i],f'{stocks[t,i]:.9f}'])
            if name=='base':
                row['busiest_arc']={'from':base['warehouses'][base['arcs'][most]['from']],
                                    'to':base['warehouses'][base['arcs'][most]['to']],
                                    'total_flow_units':float(flow[:,most].sum()),'index':most}
                slack=res.ineqlin.residual;dual=res.ineqlin.marginals
                with (out/'dual_capacity.csv').open('w',newline='',encoding='utf-8') as fp:
                    w=csv.writer(fp);w.writerow(['constraint','slack_units','marginal_rub_per_unit','saturated'])
                    for label,s,p in zip(m['names'],slack,dual):w.writerow([label,float(s),float(p),int(abs(s)<=1e-7)])
                row['saturated_arc_weeks']=[x for x,s in zip(m['names'],slack) if x.startswith('arc:') and abs(s)<=1e-7]
                h=greedy(base);row['greedy']={k:v for k,v in h.items() if k not in ('flows','inventories')}
                if h['valid']:
                    row['greedy']['gap_percent']=100*(h['objective_rub']-res.fun)/res.fun
                    save(out/'greedy_validation.json',{'valid':check(base,h['flows'],h['inventories'])['valid'],'cost':h['objective_rub']})
                # Граф складов; подписи: сумма отправок/суммарная ёмкость за 4 недели.
                G=nx.DiGraph();G.add_nodes_from(range(n))
                for k,a in enumerate(base['arcs']):G.add_edge(a['from'],a['to'],idx=k)
                pos=nx.circular_layout(G);fig,ax=plt.subplots(figsize=(11,8))
                widths=[.7+2.8*flow[:,a['idx']].sum()/max(float(flow.sum(axis=0).max()),1) for _,_,a in G.edges(data=True)]
                nx.draw_networkx_nodes(G,pos,ax=ax,node_color='#e8e8e8',edgecolors='black',node_size=900)
                nx.draw_networkx_labels(G,pos,labels=dict(enumerate(base['warehouses'])),ax=ax,font_size=11)
                nx.draw_networkx_edges(G,pos,ax=ax,arrows=True,arrowsize=13,width=widths,edge_color='#666666',connectionstyle='arc3,rad=.16',node_size=900)
                labels={(a['from'],a['to']):f'{flow[:,k].sum():.0f}/{4*a["capacity_units_per_week"]:.0f}' for k,a in enumerate(base['arcs'])}
                nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,ax=ax,font_size=6.7,rotate=False,label_pos=.35,bbox={'facecolor':'white','edgecolor':'none','pad':.1})
                ax.set_title('Total transfers / total capacity (4 weeks)');ax.axis('off');fig.tight_layout();fig.savefig(out/'warehouse_network.png',dpi=170);plt.close(fig)
                fig,ax=plt.subplots(figsize=(8.5,3.8))
                for i in range(n):ax.plot(range(1,T+1),stocks[:,i],marker='o',label=base['warehouses'][i])
                ax.set_xlabel('Week');ax.set_ylabel('End inventory, units');ax.set_xticks(range(1,T+1));ax.legend(ncol=4,fontsize=8)
                fig.tight_layout();fig.savefig(out/'inventory_by_week.png',dpi=180);plt.close(fig)
        else:save(out/f'validation_{name}.json',{'valid':False,'reason':res.message})
        runs[name]=row
        log.append(f'{name}: solver=HiGHS/highs-ds; presolve=True; status={row["status"]}; objective={row["objective_rub"]}; iterations={row["iterations"]}; message={res.message}')
    with (out/'scenarios.csv').open('w',newline='',encoding='utf8') as fp:
        w=csv.writer(fp);w.writerow(['scenario','status','objective_rub','iterations'])
        for z in runs.values():w.writerow([z['scenario'],z['status'],z['objective_rub'],z['iterations']])
    save(out/'summary.json',runs)
    (out/'solver_log.txt').write_text(f'HiGHS/scipy.optimize.linprog; Python {platform.python_version()}, SciPy {scipy.__version__}, NumPy {np.__version__}\n'+'\n'.join(log),encoding='utf8')
    print('ЛР2:',[(s,r['status'],r['objective_rub']) for s,r in runs.items()]);print('Жадная:',runs['base'].get('greedy'))
if __name__=='__main__':main()
