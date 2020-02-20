from typing import Union, Dict


class HDL2PHPS():
    def eval_str_to_dict(self,hdl):
        raise NotImplementedError()

    def after_process_dict(self,dict_:dict):
        raise NotImplementedError()

    def __call__(self, hdl:Dict):
        assert isinstance(hdl,dict)
        hdl_=self.eval_str_to_dict(hdl)
        return self.after_process_dict(hdl_)