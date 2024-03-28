from pathlib import Path
import torchaudio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ..dataset import OTFDataset
from ..data_utils import idx2motif

def save_preview(dataset, sample, dir:Path, idx, duration_sec=15):
    if len(sample) == 5:
        version, act, motif, start, end = sample
    elif len(sample) == 4:
        version, act, start, end = sample
        motif = "None"
    else:
        raise ValueError("Invalid sample format")

    fn = f"{version}_{act}"
    start_samp = start * 512
    end_samp = start_samp + (duration_sec * 22050)

    wav = dataset.wavs[fn][start_samp:end_samp]
    cqt = dataset.transform(wav.to("cuda")).squeeze(0)
    cqt = (cqt / cqt.max()).cpu()
    gt = dataset.instances_gts[fn][start:end, :]

    filename= f"{idx:03d}-{fn}-{motif}-{start_samp/22050:.1f}"

    torchaudio.save(dir / f"{filename}.wav", wav.unsqueeze(0), 22050)

    fig, ax = plt.subplots(nrows=2, figsize=(10, 10), dpi=150)
    fig.suptitle(filename)
    im = ax[0].imshow(cqt, origin="lower", aspect="auto", interpolation="bilinear", cmap="magma", norm="log", vmin=0.01)
    im2 = ax[1].imshow(gt.T, origin="upper", aspect="auto", interpolation="none", cmap="binary")
    ax[1].yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax[1].set_yticks(range(21))
    ax[1].set_yticklabels(idx2motif + ["None"])
    ax[1].tick_params(axis="y", which="minor", length=0)
    ax[1].grid(axis="y", which="minor")
    fig.tight_layout()

    fig.savefig(dir / f"{filename}.png")
    plt.close(fig)

def main():
    dataset = OTFDataset(Path("data/wav-22050"),
                        Path("data/LeitmotifOccurrencesInstances/Instances"),
                        Path("data/WagnerRing_Public/02_Annotations/ann_audio_singing"))

    selected_versions = ['Ba', 'Bh', 'Bo', 'Kr', 'Sw', 'Ha', 'Th', 'Ke', 'Le', 'Ka', 'Ne', 'Sa', 'We', 'Ja', 'Fu']
    samples = [[s for s in dataset.samples if s[0] == version] + [s for s in dataset.none_samples if s[0] == version] for version in selected_versions]

    for i, v in enumerate(tqdm(selected_versions, ascii=True)):
        dir = Path(f"tests/previews/{v}")
        dir.mkdir(parents=True, exist_ok=True)
        idx = 1
        for j, s in enumerate(tqdm(samples[i], ascii=True, leave=False)):
            if j % 50 == 0:
                save_preview(dataset, s, dir, idx)
                idx += 1

if __name__ == "__main__":
    main()