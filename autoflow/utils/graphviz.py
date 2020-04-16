#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.utils.klass import StrSignatureMixin


class ColorSelector(StrSignatureMixin):
    def __init__(self, colors):
        self.colors = colors
        self.N = len(self.colors)
        self.id = 0
        self.label2id = {}
        self.id2label = {}

    def __getitem__(self, label):
        if label=="target":
            return "#000000"
        if label in self.label2id:
            id = self.label2id[label]
        else:
            self.label2id[label] = self.id
            self.id2label[self.id] = label
            id = self.id
            self.id += 1
        return self.colors[id % self.N]
