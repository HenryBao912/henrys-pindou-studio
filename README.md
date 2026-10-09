# Henry's Pindou Studio - 给田田田的拼豆图纸

Turn a photo into a fuse-bead (拼豆 / Perler / Hama) pattern on your own computer.

**Website:** https://henrybao912.github.io/henrys-pindou-studio/

**Privacy:** the app has no networking code at all. Your photo is read from disk, processed in memory, and only written to the files you export. (The one-time `pip install` below downloads the libraries, never your photos.)

## Download

[Download the latest version](https://github.com/HenryBao912/henrys-pindou-studio/archive/refs/heads/main.zip) (same as **Code > Download ZIP** on this page), then unzip.

## Install (once)

1. Install Python 3.9 or newer from [python.org](https://www.python.org/downloads/). It includes Tkinter, the window toolkit.
   - Mac + Homebrew Python instead? Also run `brew install python-tk`.
2. Install the libraries:
   ```
   pip install pillow numpy reportlab
   ```
   Optional, to open iPhone `.HEIC` photos directly: `pip install pillow-heif`

## Run

- **Windows:** double-click `run_windows.bat`
- **Mac:** double-click `run_mac.command` (first time: right-click, then Open)
- **Any system:** `python3 henrys_pindou_studio.py` or `python3 henrys_pindou_studio.py photo.jpg`

The launchers install the libraries automatically the first time if they are missing.

## Workflow

**Tab 1 · Photo, level & crop**
1. **Open photo.**
2. Face tilted? Click **Level eyes**, then click the center of one eye and the center of the other. The photo rotates so the eyes are exactly level. Fine-tune with the Rotate slider or the ±0.5° buttons.
3. **Drag on the photo to crop.** Drag inside the box to move it. Pick a crop shape (1:1, 3:4…) if you want a fixed shape.
4. The **bead grid overlay** shows exactly where every bead falls, plus a red center line for checking symmetry. Use **Grid nudge X/Y** so each eye sits inside a bead instead of straddling two. Straddling beads are the main reason bead faces come out lopsided.

**Size** (left panel)
- Drag **Width (beads)** and watch the preview: more beads means more detail and a bigger piece.
- The readout shows beads, real size in cm and inches, how many boards, and a detail hint.
- Type a **Target width in cm** and press Set, or use the quick buttons (Keychain, 1/2/3 boards).
- Set your **bead type** (2.6 mm mini, 5 mm midi…) and **board size** (52 or 29 pegs, etc.).

**Colors**
- **Brand:** MARD 221/291, COCO, 漫漫, 盼盼, 咪小窝, 优肯/Artkal C and M, Artkal S, Perler, Perler Mini, Hama Midi, Hama Mini.
- **Max colors:** 8 to 20 is the sweet spot.
- **Clean edges** (default) picks each bead's dominant color, avoiding the grey fringe that averaging creates. **Smooth shading** averages instead.
- **Face details** (0 to 3) stops eyes, brows and lips from disappearing. Use 3 for small patterns.
- **Only use these codes:** the colors you own, e.g. `A1 B3 H7 H2` (spaces or commas). **Never use these codes:** colors to leave out. Codes are per brand; type them as the app shows them (MARD uses `A1`, not `A01`).

**Background, cleanup, outline**
- **Remove plain background** flood-fills from the edges and also removes the pale halo around the subject.
- **Cleanup** merges random single beads that are just photo noise.
- **Add outline** gives a crisp edge and ties loose bits together.

**Tab 2 · Pattern & edit**
- Views: **Beads** (realistic preview), **Flat**, **Codes**. Hover any bead to see its code.
- **Show: Grid lines / Board edges / Weak spots** switch those overlays on or off in the preview.
- **Red circles** mark spots where parts touch only at a corner. Those don't fuse well and can snap after ironing, so add a bead beside each.
- The right panel warns if the design falls apart into separate pieces.
- Edit with **Paint / Erase / Fill / Pick**. Turn on **Mirror** to fix both eyes symmetrically. Right-click erases, Ctrl/Cmd+Z undoes.
- Click a color in the list to paint with it. **Highlight selected color** dims everything else, which helps while placing beads.

## Export

**Export pattern…** writes five files:

| File | Use it for |
|---|---|
| `*_true_size.pdf` | Print at **100% / Actual size** (turn off "Fit to page"). Check that the ruler measures exactly 10 cm, then slide the page under a clear pegboard and place beads right on top. |
| `*_chart.pdf` | Overview page with legend and bead counts, then one page per board with a color code in every cell. **Chart** option: one page per board, whole pattern on one page, or large print. |
| `*_chart.png` | The whole chart as one image for your phone or iPad. |
| `*_preview.png` | What the finished piece will look like. |
| `*_colors.csv` | Shopping list: count per color plus ~10% spare. |

Paper: Letter or A4.

## What makes a good 拼豆 pattern

- One clear subject, plain background, even light. Busy photos turn into noise.
- Crop tight: everything outside the crop costs beads and detail.
- Size guide: 20 to 30 beads wide = icon/keychain, 40 to 60 = faces read well, 80+ = photo-like. Mini 2.6 mm beads give about twice the detail per cm of 5 mm beads.
- Fewer colors read better and are easier to sort.
- Screen colors approximate real beads. Hold your physical color card (色卡) next to the screen.
- Corner-only joints and floating bits break after ironing, so fix the red circles before you start.

## Command line (no window)

```
python3 henrys_pindou_studio.py photo.jpg --export out/cat --width 52 --palette mard221 --remove-bg --outline
```

Palette keys: `mard221 mard291 coco291 manman278 panpan289 mixiaowo290 artkalc197 artkalm221 artkals perler perlermini hama hamamini`. Run with `--help` for every option.

## Credits

Bead color data comes from the open datasets [HansBug/pindou-color-data](https://github.com/HansBug/pindou-color-data) and [maxcleme/beadcolors](https://github.com/maxcleme/beadcolors), as packaged by [HansBug/pypindou](https://github.com/HansBug/pypindou) (Apache-2.0). Color matching uses CIEDE2000, a perceptual color-difference formula.
