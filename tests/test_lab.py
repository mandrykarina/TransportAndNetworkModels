import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generate import make_data
from model import build,solve,scenario
from verify import check
from heuristic import greedy
class TestWarehouses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=json.loads((ROOT/'data'/'instance.json').read_text(encoding='utf8'))
    def test_seed_and_scale(self):
        self.assertEqual(self.d,make_data());self.assertEqual((len(self.d['warehouses']),self.d['weeks']),(8,4))
    def test_base_solution(self):
        m=build(self.d);r=solve(m);self.assertTrue(r.success)
        T,n,E=m['weeks'],m['n'],m['e'];v=check(self.d,r.x[:T*E].reshape(T,E),r.x[T*E:].reshape(T,n))
        self.assertTrue(v['valid']);self.assertAlmostEqual(v['objective_rub'],r.fun,places=5)
    def test_scenarios_and_closed_route(self):
        m=build(self.d);r=solve(m);T,E=m['weeks'],m['e'];busy=int(np.argmax(r.x[:T*E].reshape(T,E).sum(axis=0)))
        for kind in ('closure','congestion','demand_plus_20','stress'):
            d=scenario(self.d,kind,busy);mm=build(d);res=solve(mm)
            if res.success:
                f=res.x[:T*E].reshape(T,E);stock=res.x[T*E:].reshape(T,m['n'])
                self.assertTrue(check(d,f,stock)['valid'])
                if kind=='closure':self.assertTrue(np.all(np.abs(f[:,busy])<1e-7))
    def test_flow_balance_violation_detected(self):
        m=build(self.d);r=solve(m);T,n,E=m['weeks'],m['n'],m['e']
        f=r.x[:T*E].reshape(T,E).copy();stock=r.x[T*E:].reshape(T,n).copy();stock[0,0]+=1
        self.assertFalse(check(self.d,f,stock)['valid'])
    def test_greedy_valid_and_not_better_than_lp(self):
        result=greedy(self.d);self.assertTrue(result['valid'],result)
        optimum=solve(build(self.d));self.assertGreaterEqual(result['objective_rub']+1e-6,optimum.fun)
    def test_binding_arcs_exist(self):
        m=build(self.d);r=solve(m)
        self.assertGreater(np.sum([(name.startswith('arc:') and abs(s)<1e-7) for name,s in zip(m['names'],r.ineqlin.residual)]),0)
if __name__=='__main__':unittest.main()
