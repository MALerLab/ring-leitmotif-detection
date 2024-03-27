from pathlib import Path
import pandas as pd

occurrences_path = Path('../data/LeitmotifOccurrencesInstances/Occurrences')
fns = sorted(list(occurrences_path.glob("*.csv")))

idx2motif = ['Nibelungen',
    'Ring',
    'Nibelungenhass',
    'Mime',
    'Ritt',
    'Waldweben',
    'Waberlohe',
    'Horn',
    'Geschwisterliebe',
    'Schwert',
    'Jugendkraft',
    'Walhall-b',
    'Riesen',
    'Feuerzauber',
    'Schicksal',
    'Unmuth',
    'Liebe',
    'Siegfried',
    'Mannen',
    'Vertrag']
motif2idx = {x: i for i, x in enumerate(idx2motif)}

table = [[0 for _ in fns] for _ in idx2motif]

for i, fn in enumerate(fns):
    csv = list(pd.read_csv(fn, sep=";").itertuples(index=False, name=None))
    for row in csv:
        table[motif2idx[row[0]]][i] += 1

df = pd.DataFrame(table, 
                  columns=['A', 'B-1', 'B-2', 'B-3', 'C-1', 'C-2', 'C-3', 'D-0', 'D-1', 'D-2', 'D-3'], 
                  index=idx2motif)

df.to_csv('occurrence_stats.csv')