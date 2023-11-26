from pathlib import Path
import pandas as pd

instance_fns = sorted(list(Path("LeitmotifOccurrencesInstances/Instances/").rglob("*.csv")))
instances = []
for fn in instance_fns:
    version = fn.parent.name
    act = fn.stem
    df = pd.read_csv(fn, sep=";")
    for row in df.iterrows():
        motif = row[1]["Motif"]
        start_sec = row[1]["StartSec"]
        end_sec = row[1]["EndSec"]
        instances.append((version, act, motif, start_sec, end_sec))
df = pd.DataFrame(instances, columns=['Version', 'Act', 'Motif', 'StartSec', 'EndSec'])
df.to_csv("instances_merged.csv", index=False)

occurrence_fns = sorted(list(Path("LeitmotifOccurrencesInstances/Occurrences").rglob("*.csv")))
occurrences = []
for fn in occurrence_fns:
    act = fn.stem
    df = pd.read_csv(fn, sep=";")
    for row in df.iterrows():
        motif = row[1]["Motif"]
        start_measure = row[1]["StartMeasure"]
        end_measure = row[1]["EndMeasure"]
        occurrences.append((act, motif, start_measure, end_measure))
df = pd.DataFrame(occurrences, columns=['Act', 'Motif', 'StartMeasure', 'EndMeasure'])
df.to_csv("occurrences_merged.csv", index=False)