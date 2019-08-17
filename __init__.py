# -*- coding: utf-8 -*-
# copyright Dodd Gray 2019

try:
    from instrumental import u, Q_
except:
    from pint import UnitRegistry
    u = UnitRegistry()
    Q_ = u.Quantity
