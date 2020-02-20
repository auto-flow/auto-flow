from typing import List

from ConfigSpace.configuration_space import Configuration

from autopipeline.hdl.smac import _decode
from autopipeline.php2dhp.base import PHP2DHP


class SmacPHP2DHP(PHP2DHP):
    def set_kv(self, dict_: dict, key_path: list, value):
        tmp = dict_
        for i, key in enumerate(key_path):
            if i != len(key_path) - 1:
                if key not in tmp:
                    tmp[key] = {}
                tmp = tmp[key]
        tmp[key_path[-1]] = value

    def split_key(self,key,token="/",ignore=("[","]"))->List[str]:
        L=len(key)
        stack=0
        res=[]
        cursor=""
        for i,e in enumerate(key):
            if e == ignore[0]:
                stack+=1
            if e==token and stack==0:
                res.append(cursor)
                cursor=""
            else:
                cursor=cursor+e
            if e==ignore[1]:
                stack-=1
        res.append(cursor)
        return res

    def convert(self, php:Configuration):
        dict_=php.get_dictionary()
        ret={}
        for k,v in dict_.items():
            if isinstance(v,str):
                v=_decode(v)
            self.set_kv(ret,self.split_key(k),v)
        return ret

if __name__ == '__main__':
    d = {1: {2: {4: 6}}}
    SmacPHP2DHP().set_kv(d, [1, 2, 3], 4)
    print(d)
    print(SmacPHP2DHP().split_key("[1/2]/3/4"))
