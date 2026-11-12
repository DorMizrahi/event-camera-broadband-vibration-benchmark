"""Final compact, reproducible figures for VIBROENGINEERING 2027 R1.17.

Figure 1: actual rendered marker frame, recorded ON/OFF events and ROI,
          and the two algorithmic analysis paths.
Figure 2: actual acceleration time history in experiment_030 and distributions
          of absolute acceleration NRMSE across all 30 original experiments.

All plotted data are loaded from the original archived project data; the flow
panel in Figure 1 is a schematic of the actual processing described in the paper.

To reproduce, place this script beside the original three archives, then run:
    python Figure1_Figure2_FINAL_generation.py
Dependencies: numpy, pandas, matplotlib, opencv-python, h5py.
"""
from pathlib import Path
import io
import json
import tempfile
import zipfile

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EVENTS_ARCHIVE = ROOT / 'Broadband_Event_Camera_Vibration_Dataset_Minimal_v1(1).zip'
BASELINE_ARCHIVE = ROOT / 'baseline_reproduction.zip'
ANALYSIS_ARCHIVE = ROOT / 'VIBROENGINEERING2027_REVIEWER_ANALYSIS.zip'
for archive in (EVENTS_ARCHIVE, BASELINE_ARCHIVE, ANALYSIS_ARCHIVE):
    if not archive.is_file():
        raise FileNotFoundError(f'Missing original archive: {archive.name}')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10.6,
    'axes.labelsize': 11,
    'axes.titlesize': 11.6,
    'xtick.labelsize': 9.7,
    'ytick.labelsize': 9.7,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Only original, archived observations are used.
prefix = 'Broadband_Event_Camera_Vibration_Dataset_Minimal_v1/experiments/experiment_025/'
with zipfile.ZipFile(EVENTS_ARCHIVE) as archive:
    metadata = json.loads(archive.read(prefix + 'metadata.json'))
    gt = pd.read_csv(io.BytesIO(archive.read(prefix + 'ground_truth.csv')))
    with tempfile.TemporaryDirectory() as temp_dir:
        td = Path(temp_dir)
        h5_file = td / 'events.h5'
        h5_file.write_bytes(archive.read(prefix + 'events.h5'))
        with h5py.File(h5_file, 'r') as h5:
            events = h5['events'][:]
        video_file = td / 'video.mp4'
        video_file.write_bytes(archive.read(prefix + 'video.mp4'))
        video = cv2.VideoCapture(str(video_file))
        video.set(cv2.CAP_PROP_POS_FRAMES, round(0.19895 * metadata['video_fps']))
        success, bgr_frame = video.read()
        video.release()
        if not success:
            raise RuntimeError('Unable to decode source frame from experiment_025')
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

with zipfile.ZipFile(BASELINE_ARCHIVE) as archive:
    roi = json.loads(archive.read('fresh/experiment_025/selected_roi.json'))
    kinematics = pd.read_csv(io.BytesIO(archive.read(
        'fresh/experiment_030/displacement_velocity_acceleration.csv')))
with zipfile.ZipFile(ANALYSIS_ARCHIVE) as archive:
    experiments = pd.read_csv(io.BytesIO(archive.read(
        'VIBROENGINEERING2027_REVIEWER_ANALYSIS/results/experiment_level_summary.csv')))
assert len(experiments) == 30, 'Expected the 30 archived benchmark experiments'

C_ON = 'C0'
C_OFF = 'C1'
C_FOURIER = 'C0'
C_GRADIENT = 'C1'

# -------------------- FIGURE 1 -------------------------------------------
# Two compact data panels plus one readable processing schematic.
fig = plt.figure(figsize=(9.25, 3.64), facecolor='white')
grids = GridSpec(2, 2, figure=fig, height_ratios=[1.35, 1],
                 width_ratios=[1.0, 1.55], left=.080, right=.985,
                 top=.924, bottom=.085, hspace=.46, wspace=.28)
ax_frame = fig.add_subplot(grids[0, 0])
ax_events = fig.add_subplot(grids[0, 1])

ax_frame.imshow(gray_frame, cmap='gray', origin='upper',
                extent=(-.5, 239.5, 179.5, -.5), vmin=0, vmax=255,
                interpolation='nearest')
ax_frame.add_patch(Rectangle((roi['x_min'], roi['y_min']),
                             roi['width'], roi['height'], fill=False,
                             edgecolor='white', linestyle='--', lw=1.5))
ax_frame.set(xlim=(96, 148), ylim=(108, 71),
             xlabel='Image x (px)', ylabel='Image y (px)')
ax_frame.set_title('(a) Rendered marker · 198.95 ms', loc='left', pad=6, weight='semibold')
ax_frame.grid(ls=':', alpha=.14)

segment = events[(events[:, 0] >= 198950) & (events[:, 0] < 199200)]
for polarity, symbol, name, color in ((1, '.', 'ON', C_ON), (0, 'x', 'OFF', C_OFF)):
    q = segment[:, 3] == polarity
    ax_events.scatter(segment[q, 1], segment[q, 2], s=10, marker=symbol,
                      color=color, alpha=.75, linewidths=.7,
                      label=name, rasterized=True)
truth = gt[(gt.t_sec >= .19895) & (gt.t_sec < .19920)]
ax_events.plot(truth.x_px + 120, truth.y_px + 90,
               color='black', lw=1.5, label='GT path')
ax_events.add_patch(Rectangle((roi['x_min'], roi['y_min']),
                              roi['width'], roi['height'], fill=False,
                              edgecolor='black', ls='--', lw=1.5))
ax_events.annotate('Offline ROI',
                   xy=(roi['x_max'], roi['y_min'] + 1),
                   xytext=(139, 73.4), ha='left', fontsize=8.7,
                   arrowprops=dict(arrowstyle='->', lw=.9, color='black'))
ax_events.set(xlim=(97, 151), ylim=(109, 70),
              xlabel='Image x (px)', ylabel='Image y (px)')
ax_events.set_title('(b) Recorded ON/OFF events · 250 µs', loc='left', pad=6,
                    weight='semibold')
ax_events.legend(loc='lower center', bbox_to_anchor=(.55, .008), ncol=3,
                 columnspacing=1.25, handlelength=1.1, handletextpad=.33,
                 framealpha=.95, facecolor='white', edgecolor='.8',
                 fontsize=8.5, borderpad=.28)
ax_events.grid(ls=':', alpha=.13)

# This is a workflow schematic; it does not depict measured/synthetic signal data.
flow = fig.add_subplot(grids[1, :]); flow.set_xlim(0, 1); flow.set_ylim(0, 1)
flow.axis('off')
flow.text(.005, .97, '(c) Processing workflow', fontsize=11.7,
          weight='semibold', va='top')

def draw_box(x, y, w, h, label, size=9.6, dashed=False):
    shape = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.006,rounding_size=0.012',
                           facecolor='#f8fafc', edgecolor='#29303a', linewidth=1.0,
                           linestyle='--' if dashed else '-')
    flow.add_patch(shape)
    flow.text(x + w / 2, y + h / 2, label, fontsize=size,
              ha='center', va='center', linespacing=1.06)

def draw_arrow(start, end):
    flow.add_patch(FancyArrowPatch(start, end, arrowstyle='->',
                                   mutation_scale=11, lw=1.0,
                                   color='#29303a'))

draw_box(.009, .24, .127, .34, 'Event record\n(0.2 s)', 9.5)
draw_box(.157, .24, .140, .34, 'Offline ROI\n(full record)', 9.2)
draw_box(.322, .24, .143, .34, '50-µs ON/OFF\ncentroids', 9.0)
draw_box(.512, .58, .245, .26, 'Spectral (ON)\nSG → peaks / Chirp', 9.1)
draw_box(.512, .075, .245, .30, 'Kinematics (ON)\nbandpass → derivatives', 8.8)
draw_box(.797, .24, .191, .34, 'Ground truth\n(evaluation only)', 8.9, True)
draw_arrow((.136, .41), (.157, .41))
draw_arrow((.297, .41), (.322, .41))
draw_arrow((.465, .43), (.512, .71))
draw_arrow((.465, .39), (.512, .23))
draw_arrow((.757, .71), (.797, .48))
draw_arrow((.757, .23), (.797, .38))

for ext in ('png', 'pdf'):
    fig.savefig(ROOT / f'Figure1_FINAL.{ext}', dpi=330, facecolor='white')
plt.close(fig)

# -------------------- FIGURE 2 -------------------------------------------
# Remove the relative-RMSE subplot (already in the prose/table) to retain the
# two evidence panels the reviewer explicitly requested.
fig = plt.figure(figsize=(9.25, 3.12), facecolor='white')
grids = GridSpec(1, 2, figure=fig, width_ratios=[1.18, 1.0],
                 left=.100, right=.985, top=.86, bottom=.19, wspace=.35)
a, b = fig.add_subplot(grids[0, 0]), fig.add_subplot(grids[0, 1])
segment = kinematics[(kinematics.t_sec >= .08) & (kinematics.t_sec <= .09)]
for column, name, linestyle, color, line_width in (
    ('x_acceleration_gt', 'Ground truth', '-', 'C2', 1.9),
    ('x_acceleration_frequency_domain', 'Fourier', '--', C_FOURIER, 1.15),
    ('x_acceleration_time_domain', 'Gradient', '-', C_GRADIENT, 1.3),
):
    a.plot(1000 * segment.t_sec, segment[column] / 1e6,
           label=name, ls=linestyle, color=color, lw=line_width)
a.set(xlabel='Time (ms)', ylabel='Acceleration (10⁶ px/s²)', xlim=(80, 90))
a.set_xticks([80, 82, 84, 86, 88, 90])
a.set_title('(a) Reference and reconstructed acceleration',
            loc='left', pad=6, weight='semibold')
a.text(.02, .025, 'experiment_030 · 2.5 px', transform=a.transAxes,
       fontsize=8.4, va='bottom', ha='left',
       bbox={'boxstyle': 'round,pad=.16', 'facecolor':'white',
             'edgecolor':'.85', 'alpha':.93})
a.grid(alpha=.14)
a.legend(loc='upper right', fontsize=8.6,
         framealpha=.94, ncol=1, borderpad=.25)

amplitudes = (11.0, 6.0, 2.5)
xloc = np.arange(3)
for idx, (column, name, color) in enumerate((
    ('acceleration_fd_nrmse', 'Fourier', C_FOURIER),
    ('acceleration_td_nrmse', 'Gradient', C_GRADIENT),
)):
    offset = -.18 if idx == 0 else .18
    grouped = [experiments.loc[experiments.amplitude == amp, column].dropna().to_numpy()
               for amp in amplitudes]
    box = b.boxplot(grouped, positions=xloc + offset, widths=.28,
                    manage_ticks=False, showfliers=False, patch_artist=True,
                    medianprops={'color': '#171717', 'lw': 1.45},
                    boxprops={'linewidth': 1.0, 'edgecolor': color},
                    whiskerprops={'linewidth': 1.0, 'color': color},
                    capprops={'linewidth': 1.0, 'color': color})
    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(.18)
    for j, values in enumerate(grouped):
        # Deterministic visual offsets only; data values remain unchanged.
        jitter = np.linspace(-.055, .055, len(values)) if len(values) > 1 else [0]
        b.scatter(j + offset + jitter, values, c=color, s=19, alpha=.77,
                  zorder=3, edgecolor='white', linewidth=.3,
                  label=name if j == 0 else None)
b.axhline(1, ls='--', lw=1.05, color='gray', alpha=.75)
b.set(xticks=xloc, xticklabels=['11', '6', '2.5'],
      xlabel='Nominal displacement (px)',
      ylabel='Acceleration NRMSE', ylim=(0, 3.24))
b.set_title('(b) Acceleration NRMSE · 30 cases',
            loc='left', pad=6, weight='semibold')
b.legend(loc='upper left', ncol=2, frameon=False, fontsize=8.5,
         columnspacing=.75, handletextpad=.35)
b.grid(axis='y', alpha=.14)

for ext in ('png', 'pdf'):
    fig.savefig(ROOT / f'Figure2_FINAL.{ext}', dpi=330, facecolor='white')
plt.close(fig)

print('Rebuilt Figure1_FINAL and Figure2_FINAL from the three original archives.')
