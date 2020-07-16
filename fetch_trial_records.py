#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.resource_manager.base import ResourceManager
import json
from pathlib import Path

rm=ResourceManager()
rm.init_trial_table()
Trial=rm.TrialModel
records=Trial.select(Trial.config,Trial.loss).where(Trial.experiment_id==113).dicts()
records=list(records)
Path("trial.json").write_text(json.dumps(records))