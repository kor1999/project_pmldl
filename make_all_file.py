import pandas as pd

import glob
paths = glob.glob('data1/*')
paths.extend(glob.glob('data2/*'))
paths.extend(glob.glob('data3/*'))

csvs = [pd.read_csv(path) for path in paths]
total = None
for csv in csvs:
    if total is None:
        total = csv
    else:
        total = pd.concat((total, csv))
print(len(total))

total.columns = ['id', 'content']
total.to_csv('all.csv', index=False)