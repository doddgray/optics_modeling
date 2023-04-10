from pint import UnitRegistry
u   =   UnitRegistry(autoconvert_offset_to_baseunit=True)
Q_  =   u.Quantity

"""
By default use 'short pretty' ('~P') format for unit strings,
like 'm²/s²' rather than 'meter**2/second**2' ('~D')
or 'm**2/s**2' ('~C'). The '~' modifier turns on the 'short'
format, eg. 'm' instead of 'meter'.
"""
u.default_format = '~P'

"""
Evaluate the magnitude and unit format specs independently, 
such that with a global default of ureg.default_format = ".3f" 
and f"{q:P}" the format that will be used is ".3fP". 
"""
u.separate_format_defaults = True
