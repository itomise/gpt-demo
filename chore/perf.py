import pandas as pd
import math

avgs = pd.read_csv("./data/full_text_avg.csv")

sum = 0.0
_max = 0.0
_min = avgs['avg'][0]
    
for avg in avgs['avg']:
    _max = max(avg, _max)
    _min = min(avg, _min)
    sum += avg
    
print(sum / len(avgs['avg']))
print(_max)
print(_min)