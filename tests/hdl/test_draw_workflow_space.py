import os
from pathlib import Path

from autoflow import DataManager, HDL_Constructor
from autoflow.datasets import load
from autoflow.tests.base import LocalResourceTestCase


class TestHDL_Visualize(LocalResourceTestCase):
    def test_draw_only_estimator(self):
        name = "test_draw_workspace"
        train_df = load("qsar")
        remain_cols = list(train_df.columns)
        remain_cols.remove("target")
        column_descriptions = {
            "num": remain_cols,
            "target": "target"
        }
        data_manager = DataManager(self.mock_resource_manager, train_df, column_descriptions=column_descriptions)
        hdl_constructor = HDL_Constructor(DAG_workflow={
            "num->target": ["lightgbm", "catboost"]
        })
        hdl_constructor.run(data_manager)
        hdl_df = hdl_constructor.get_hdl_dataframe()
        Path(f"{name}.html").write_text(hdl_df.to_html())
        hdl_df.to_excel(f"{name}.xlsx")
        # pip install openpyxl
        print(hdl_df)
        graph = hdl_constructor.draw_workflow_space()
        open(f"{name}.gv", "w+").write(graph.source)
        cmd = f'''dot -Tpng -Gsize=9,15\! -Gdpi=300 -o{name}.png {name}.gv'''
        os.system(cmd)

