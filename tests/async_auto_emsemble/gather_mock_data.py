#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import peewee as pw
from joblib import load
import seaborn as sns
import pylab as plt
import pandas as pd
from  joblib import dump

db_params = {
    "user": "tqc",
    "host": "127.0.0.1",
    "port": 5432,
}
db = pw.PostgresqlDatabase("autoflow_meta_bo", **db_params)
cursor = db.execute_sql("""
select trial_id, budget, estimator, cost_time, loss, test_loss,
       additional_info->'best_iterations' as best_iterations,
       config->'preprocessing:normed->final:__choice__' as feature_engineer,
       config->'preprocessing:combined->normed:__choice__' as encoder,
       config->'preprocessing:impute:__choice__' as imputer,
       additional_info->'cost_times' as cost_times, status,
       y_info_path
from trial 
where task_id = '439f1de1aa95d757d005c9c5c60e3b63'
order by start_time;
""")
info = list(cursor.fetchall())
head_str="trial_id,budget,estimator,cost_time,loss,test_loss,best_iterations,feature_engineer,encoder,imputer,cost_times,status,y_info_path"
head=head_str.split(",")
y_info_list=[]
for row in info:
    y_info=load(row[-1])
df = pd.DataFrame(info, columns=head)
df=df.query("status == 'SUCCESS'")
# sns.lmplot('loss', 'test_loss', df, size=8)
sns.jointplot(df['loss'], df['test_loss'], kind='reg', size=8)
plt.show()
print("pearson correlation", df[['loss', 'test_loss']].corr().values[0][1])
print("kendall correlation", df[['loss', 'test_loss']].corr("kendall").values[0][1])
print("spearman correlation", df[['loss', 'test_loss']].corr("spearman").values[0][1])
dump((info, head, y_info_list), "mock_data.bz2")
