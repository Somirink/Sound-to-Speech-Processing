

import sys
sys.stdout.reconfigure(encoding="utf-8")

import parselmouth
from parselmouth.praat import call
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict
from matplotlib.patches import Ellipse

# ========= SETTINGS =========
DATA_PATH = r"C:\Users\somir\OneDrive\Documents\OneDrive\Desktop\Tangkhul_vvowels\data"
VOWELS = ['a', 'i', 'o', 'u', 'ā']   # <-- removed 'e'

# Map TextGrid labels → vowel categories
LABEL_MAP = {
    'ʌ': 'a',
    'i:': 'i',
    'o': 'o',
    'u': 'u',
    'ʌː': 'ā'
}

# Color per vowel (removed 'e')
vowel_colors = {
    'a': 'blue',
    'i': 'brown',
    'o': 'green',
    'u': 'red',
    'ā': 'purple'
}

MAX_FORMANT = 5000
TIER_NUM = 1

# ========= FUNCTIONS =========

def extract_formants(sound, start, end, max_formant=MAX_FORMANT):
    midpoint = (start + end) / 2
    formant = call(sound, "To Formant (burg)", 0.0, 5, max_formant, 0.025, 50)
    f1 = call(formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
    f2 = call(formant, "Get value at time", 2, midpoint, "Hertz", "Linear")
    if 200 < f1 < 1200 and 500 < f2 < 3500:
        return f1, f2
    return None, None


def lobanov_normalize(df):
    df2 = df.copy()
    df2['F1_lob'] = 0.0
    df2['F2_lob'] = 0.0
    for spk, g in df.groupby('speaker'):
        f1m, f1s = g['F1'].mean(), g['F1'].std()
        f2m, f2s = g['F2'].mean(), g['F2'].std()
        idx = df2['speaker'] == spk
        df2.loc[idx, 'F1_lob'] = (df2.loc[idx, 'F1'] - f1m) / f1s
        df2.loc[idx, 'F2_lob'] = (df2.loc[idx, 'F2'] - f2m) / f2s
    return df2


def draw_sd_ellipse(F2_mean, F1_mean, F2_sd, F1_sd, ax, color, k=1.0):
    if np.isnan(F1_sd) or np.isnan(F2_sd) or F1_sd == 0 or F2_sd == 0:
        return

    ell = Ellipse(
        (F2_mean, F1_mean),
        width=F2_sd * k,
        height=F1_sd * k,
        edgecolor=color,
        facecolor='none',
        linestyle='dashed',
        linewidth=1.3,
    )
    ax.add_patch(ell)


# ========= MAIN EXTRACTION =========
data = []
speaker_vowel_counts = defaultdict(Counter)

if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(f"Folder not found: {DATA_PATH}")

for filename in os.listdir(DATA_PATH):
    if filename.endswith(".wav"):
        speaker = os.path.splitext(filename)[0]
        wav_path = os.path.join(DATA_PATH, filename)
        tg_path = wav_path.replace(".wav", ".TextGrid")

        if not os.path.exists(tg_path):
            print(f"⚠ Skipping {speaker} — no TextGrid found.")
            continue

        sound = parselmouth.Sound(wav_path)
        tg = parselmouth.read(tg_path)

        n_tiers = call(tg, "Get number of tiers")
        print(f"\n🎙 Speaker: {speaker} — {n_tiers} tiers found")

        num_intervals = call(tg, "Get number of intervals", TIER_NUM)

        for i in range(1, num_intervals + 1):
            raw_label = call(tg, "Get label of interval", TIER_NUM, i).strip().lower()
            start = call(tg, "Get start time of interval", TIER_NUM, i)
            end = call(tg, "Get end time of interval", TIER_NUM, i)

            label = LABEL_MAP.get(raw_label, raw_label)

            if label in VOWELS:
                f1, f2 = extract_formants(sound, start, end)
                if f1 is not None and f2 is not None:
                    data.append([speaker, label, f1, f2])
                    speaker_vowel_counts[speaker][label] += 1


# ========= BUILD DATAFRAME =========
df = pd.DataFrame(data, columns=['speaker', 'vowel', 'F1', 'F2'])
print(f"\n✅ Extracted {len(df)} vowel tokens.")

# ========= TOTAL COUNTS =========
overall_counts = df['vowel'].value_counts().reindex(VOWELS, fill_value=0)
print("\n📈 Total tokens per vowel:")
print(overall_counts)

# ========= SAVE RAW DATA =========
raw_csv = os.path.join(DATA_PATH, "Tangkhul_formants_raw.csv")
df.to_csv(raw_csv, index=False)

# ========= LOBANOV NORMALIZATION =========
df_norm = lobanov_normalize(df)
norm_csv = os.path.join(DATA_PATH, "Tangkhul_formants_lobanov.csv")
df_norm.to_csv(norm_csv, index=False)

# Mean & SD per vowel
stats_lob = df_norm.groupby('vowel').agg(
    F1_mean=('F1_lob', 'mean'),
    F2_mean=('F2_lob', 'mean'),
    F1_sd=('F1_lob', 'std'),
    F2_sd=('F2_lob', 'std')
)


# ========= RAW PLOT WITH COLOR ELLIPSES =========
means_raw = df.groupby('vowel')[['F1', 'F2']].mean()
sdev_raw = df.groupby('vowel')[['F1', 'F2']].std()

plt.figure(figsize=(8, 6))

for v in means_raw.index:
    row = means_raw.loc[v]
    sd = sdev_raw.loc[v]
    color = vowel_colors.get(v, "black")

    plt.scatter(row['F2'], row['F1'], color=color)
    plt.text(row['F2'], row['F1'], v, fontsize=12, ha='center')

    draw_sd_ellipse(row['F2'], row['F1'], sd['F2'], sd['F1'], plt.gca(), color)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
plt.title("Tangkhul vowels (Raw Hz) with SD ellipses")
plt.grid(True)
plt.tight_layout()
plt.show()


# ========= LOBANOV NORMALIZED PLOT =========
LABEL_DX = 0
LABEL_DY = 0.15
DOT_SIZE = 100
FONT_SIZE = 11

marker_map = {
    'i': '^',
    'o': 'D',
    'u': 'p',
    'a': 'o',
    'ā': '*'
}

plt.figure(figsize=(8, 8))

for v, row in stats_lob.iterrows():
    F1m = row['F1_mean']
    F2m = row['F2_mean']
    F1sd = row['F1_sd']
    F2sd = row['F2_sd']

    symbol = marker_map.get(v, 'o')
    color = vowel_colors.get(v, 'black')

    plt.scatter(F2m, F1m, s=DOT_SIZE, marker=symbol, color=color)
    plt.text(F2m + LABEL_DX, F1m + LABEL_DY, v, fontsize=FONT_SIZE)

    draw_sd_ellipse(F2m, F1m, F2sd, F1sd, plt.gca(), color)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel("F2 (z-score)")
plt.ylabel("F1 (z-score)")
plt.title("Lobanov-normalized vowel means")
plt.grid(False)
plt.tight_layout()
plt.show()