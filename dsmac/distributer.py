from copy import deepcopy

import ray
from dsmac.runhistory.runhistory import RunHistory
from dsmac.tae.execute_ta_run import ExecuteTARun
from frozendict import frozendict
import multiprocessing as mp

from joblib import parallel_backend,delayed,Parallel

class Distributer():
    def __init__(self,tae_runner:ExecuteTARun=None,n_jobs=1):
        self.n_jobs = self.parse_n_jobs(n_jobs)
        self.tae_runner = tae_runner

    def run(self,to_run,runhistory:RunHistory):
        n_jobs = self.n_jobs
        ans=[]
        for i in range(0, len(to_run), n_jobs):
            ret=self._run(to_run[i:i + n_jobs])
            ans.extend(ret)
        self.add_runhistory(to_run,ans,runhistory)
        return ans

    def _run(self, to_run:list)->list:
        raise NotImplementedError()

    def add_runhistory(self,to_run,ans,runhistory):
        for config,(status, cost, dur, res) in zip(to_run,ans):
            if not runhistory.get_runs_for_config(config):
                runhistory.add(
                    config,
                    cost,
                    dur,
                    status
                )
        return runhistory

    def parse_n_jobs(self,n_jobs):
        if n_jobs==0:
            return 1
        elif n_jobs<0:
            return mp.cpu_count()+1+n_jobs
        else:
            return n_jobs

@ray.remote
def ray_run_fun(tae_runner, challenger):
    status, cost, dur, res = tae_runner.start(
        config=challenger,
        instance=None)
    return (status, cost, dur, res)

class RayDistributer(Distributer):
    def __init__(self,tae_runner:ExecuteTARun=None,
                 ray_config:dict=frozendict(),n_jobs=-1):
        super(RayDistributer, self).__init__(tae_runner,n_jobs)

        ray.init(**ray_config)
        print('init ray')

    # def __del__(self):
    #     print('del ray')
    #     ray.shutdown()

    def _run(self,to_run):
        ans=[ray_run_fun.remote(self.tae_runner, item) for item in to_run]
        ray_ans= ray.get(ans)   # (status, cost, dur, res)
        return ray_ans

def mp_run_fun(tae_runner,config,inst): #,q
    ans=tae_runner.start(config,inst)
    return ans
    # q.put((config,ans))

class MultiProcessDistributer(Distributer):
    def _run(self,to_run):
        processes = []
        q=mp.Queue()
        for config in to_run:
            p = mp.Process(target=mp_run_fun, args=(self.tae_runner,config,None,q))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        results = [q.get() for j in processes]
        anss=[result[1] for result in results]
        return anss

global g_tae_runner

def joblib_run_fun( challenger):
    status, cost, dur, res = g_tae_runner.start(
        config=challenger,
        instance=None)
    return (status, cost, dur, res)


class JoblibDistributer(Distributer):
    def __init__(self,*args,**kwargs):
        super(JoblibDistributer, self).__init__(*args,**kwargs)
        global g_tae_runner
        g_tae_runner=self.tae_runner

    def _run(self, to_run:list) ->list:
        with parallel_backend(backend="multiprocessing",n_jobs=-1):
            # ans=Parallel()(delayed(self.tae_runner.start)(x,None) for x in to_run)
            ans=Parallel()(delayed(mp_run_fun)(deepcopy(self.tae_runner),x,None) for x in to_run)
        return ans

class SingleDistributer(Distributer):
    def __init__(self,*args,**kwargs):
        super(SingleDistributer, self).__init__(*args,**kwargs)
        self.n_jobs=1

    def _run(self,to_run):
        ans=[self.tae_runner.start(item,None) for item in to_run]
        return ans