#!/usr/bin/python3
import sys
from collections import namedtuple


def to_time(num):
    if num > 10000:
        return str(round(num/1000)) + 'ms'
    elif num > 1000:
        return str(round(num/1000, 1)) + 'ms'
    else:
        return str(num) + 'us'


keys = ['name', 'total', 'average', 'total_own', 'average_own', 'counter']
keys_dict = {name: i for i, name in enumerate(keys)}
Entry = namedtuple('Entry', keys)
entries = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        entry = line.split()
        entry[1:] = list(map(int, entry[1:]))
        entries.append(Entry(*entry))

if len(sys.argv) >= 3:
    sort_key = keys_dict[sys.argv[2]]
else:
    sort_key = 1 # total
entries.sort(key=lambda x: x[sort_key], reverse=True)
all_time = sum((entry.total_own for entry in entries))
for entry in entries:
    percentage = '(' + str(round(entry.total / all_time * 100, 1)) + '%)'
    percentage = '{:<7}'.format(percentage)
    percentage_own = '(' + str(round(entry.total_own / all_time * 100, 1)) + '%)'
    percentage_own = '{:<7}'.format(percentage_own)
    line = list(entry)
    line[1:5] = list(map(to_time, line[1:5]))
    line[0] = "{:<20}".format(line[0])
    line[1:] = list(map(lambda x: "{:>7}".format(x), line[1:]))
    line = line[:2] + [percentage] + line[2:4] + [percentage_own] + line[4:]
    print(*line, sep=' ')
