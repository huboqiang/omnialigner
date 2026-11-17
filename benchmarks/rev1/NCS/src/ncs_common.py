"""Strict aggregation used by the frozen benchmark."""
import math
METRICS = ['PCC', 'CosSim', 'SSIM', 'MI', 'Expression_MSE_native_ASSD_bygrid', 'Accuracy']

def strict_mean(values,expected):
    if len(values)!=expected:raise ValueError('Incomplete chain/panel')
    return math.fsum(values)/expected if all(v is not None and math.isfinite(v)for v in values) else None
