import random

def sample_instance_intervals(instances, duration, total_duration):
    """
    Generates a list of intervals where a leitmotif occurs.\n
    * instances: list of tuples (Motif, StartSec, EndSec)
    * duration: Desired duration of sampled intervals(in seconds).
    * total_duration: The total duration of the recording(in seconds).\n
    If an instance is shorter than the desired duration, context is added randomly before and after the instance.\n
    If an instance is longer than the desired duration, it is randomly cropped.
    """
    intervals = []
    for instance in instances:
        start = instance[1]
        end = instance[2]
        instance_duration = end - start
        if instance_duration < duration:
            context = duration - instance_duration
            room_before = start
            room_after = total_duration - end - 0.02
            if room_before < context:
                context_before = min(room_before, context)
                context_after = context - context_before
            elif room_after < context:
                context_after = min(room_after, context)
                context_before = context - context_after
            else:
                context_before = round(random.uniform(0, context), 4)
                context_after = context - context_before
            start = round(start - context_before, 4)
            end = round(end + context_after, 4)
        elif instance_duration > duration:
            start = round(random.uniform(start, end - duration), 4)
            end = round(start + duration, 4)
        intervals.append((instance[0], start, end))
    return intervals

def generate_non_overlapping_intervals(instances, total_duration):
    """
    Generates a list of non-overlapping intervals from a list of leitmotif instances.\n
    * instances: list of tuples (Motif, StartSec, EndSec)
    * total_duration: The total duration of the audio file(in seconds).
    """
    instances.sort(key=lambda x: x[1]) # Should change
    intervals = []
    last_end = 0
    for instance in instances:
        start = instance[1]
        end = instance[2]
        if start > last_end:
            intervals.append((last_end, start))
        last_end = end
    if last_end < total_duration:
        intervals.append((last_end, total_duration))
    return intervals

def sample_non_overlapping_interval(intervals, duration):
    """
    Given a list of non-overlapping intervals, randomly samples a time interval where no leitmotif occurs.\n
    * intervals: list of tuples (start, end)
    * duration: The duration of the interval to be sampled(in seconds).
    """
    valid_intervals = [x for x in intervals if x[1] - x[0] >= duration]
    if len(valid_intervals) == 0:
        return None
    selected_interval = random.choice(valid_intervals)
    start = round(random.uniform(selected_interval[0], selected_interval[1] - duration), 3)
    end = round(start + duration, 3)
    return (start, end)

def get_binary_f1(pred, gt, threshold):
    """
    Returns (f1, precision, recall) for binary classification.\n
    pred and gt must have the same shape.\n
    """
    tp = ((pred > threshold) & (gt == 1)).sum().item()
    if tp == 0:
        tp = 0.0001
    fp = ((pred > threshold) & (gt == 0)).sum().item()
    fn = ((pred <= threshold) & (gt == 1)).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def get_tp_fp_fn(pred, gt, threshold):
    """
    Returns (f1, precision, recall) for binary classification.\n
    pred and gt must have the same shape.\n
    """
    tp = ((pred > threshold) & (gt == 1)).sum().item()
    if tp == 0:
        tp = 0.0001
    fp = ((pred > threshold) & (gt == 0)).sum().item()
    fn = ((pred <= threshold) & (gt == 1)).sum().item()
    return tp, fp, fn

def get_multiclass_acc(pred, gt):
    """
    Returns accuracy for multiclass classification.\n
    * pred: (batch, num_classes, length)
    * gt: (batch, length)
    """
    pred = pred.argmax(dim=1)
    return (pred == gt).sum().item() / (pred.shape[0] * pred.shape[1])

idx2motif = [
    'Nibelungen',
    'Ring',
    'Nibelungenhass',
    # 'Mime',
    'Ritt',
    'Waldweben',
    'Waberlohe',
    'Horn',
    # 'Geschwisterliebe',
    'Schwert',
    # 'Jugendkraft',
    'Walhall-b',
    # 'Riesen',
    'Feuerzauber',
    # 'Schicksal',
    'Unmuth',
    # 'Liebe',
    'Siegfried',
    # 'Mannen',
    'Vertrag'
]

motif2idx = {x: i for i, x in enumerate(idx2motif)}

motif2id = {
    'Nibelungen': 'L-Ni',
    'Ring': 'L-Ri',
    'Nibelungenhass': 'L-NH',
    'Mime': 'L-Mi',
    'Ritt': 'L-RT',
    'Waldweben': 'L-Wa',
    'Waberlohe': 'L-Wa',
    'Horn': 'L-Ho',
    'Geschwisterliebe': 'L-Ge',
    'Schwert': 'L-Sc',
    'Jugendkraft': 'L-Ju',
    'Walhall-b': 'L-WH',
    'Riesen': 'L-RS',
    'Feuerzauber': 'L-Fe',
    'Schicksal': 'L-SK',
    'Unmuth': 'L-Un',
    'Liebe': 'L-Li',
    'Siegfried': 'L-Si',
    'Mannen': 'L-Ma',
    'Vertrag': 'L-Ve'
}

id2version = {
    'Ba': 'Barenboim1991',
    'Bh': 'Bohm1967',
    'Bo': 'Boulez1980',
    'Fu': 'Furtwangler1953',
    'Ha': 'Haitink1988',
    'Ja': 'Janowski1980',
    'Ka': 'Karajan1966',
    'Ke': 'KeilberthFurtw1952',
    'Kr': 'Krauss1953',
    'Le': 'Levine1987',
    'Ne': 'Neuhold1993',
    'Sa': 'Sawallisch1989',
    'So': 'Solti1958',
    'Sw': 'Swarowsky1968',
    'Th': 'Thielemann2011',
    'We': 'Weigle2010'
}

version2idx = {x: i for i, x in enumerate(id2version.keys())}