#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import math


def get_int_length(number):
    Length = 0
    while number != 0:
        Length += 1
        number = number // 10    #关键，整数除法去掉最右边的一位
    return Length


def float_gcd(a, b):
    def is_int(x):
        return not bool(int(x) - x)

    base = 1
    while not (is_int(a) and is_int(b)):
        a *= 10
        b *= 10
        base *= 10
    return math.gcd(int(a), int(b)) / base