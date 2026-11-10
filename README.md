# Sanborn building labeling

This folder is everything you need to label buildings on old Sanborn fire insurance maps. The labels
train a computer model that finds every building on a map by itself, so the work you do here decides how
good that model gets.

You will not draw buildings by hand. A program has already cut each map sheet into many small colored
pieces, called **regions**, along every line it could find. Most buildings are cut into 1 to 4 regions.
Your job, in a web page that runs on your own computer, is to:

1. **merge** the regions that belong to one building (click, Shift+click),
2. **drop** regions that are not buildings (circles, symbols, stains, text),
3. **cut** a region in two in the rare place where the program missed a wall,
4. **check** every building, then run one command (**dissolve**) that turns your work into the final labels,
5. **push** (upload) your work with Git so it reaches the project.

No programming is needed. You will type a few commands that are written out below; copy them exactly.

Contents: [The sheets](#the-sheets) · [Notes per sheet](#notes-per-sheet) · [One-time setup](#one-time-setup) · [Opening a terminal](#opening-a-terminal-in-the-project-folder) ·
[Labeling a sheet](#labeling-a-sheet) · [Dissolve and check](#dissolve-and-check-the-result) ·
[Sending your work](#sending-your-work-git) · [Problems](#problems-and-fixes) · [For the project lead](#for-the-project-lead)

---

## The sheets

There are 8 sheets. **Label only the sheet(s) assigned to you**, so two people never change the same
sheet (Git cannot combine two people's work on one sheet).

| Sheet id | Place, year | Regions | Assigned to |
|---|---|---|---|
| `sanborn00985_005_0006` | Denver CO, 1903 (test sheet) | 409 | |
| `sanborn00813_008_0006` | San Francisco CA, 1899, sheet 6 | 626 | |
| `sanborn00813_008_0007` | San Francisco CA, 1899, sheet 7 | 1652 | |
| `sanborn01500_003_0012` | Savannah GA, 1898, sheet 12 | 871 | |
| `sanborn00799_001_0010L` | Sacramento CA, 1895, sheet 10 (left half) | 1719 | |
| `sanborn08891_003_0008` | Salt Lake City UT, 1898, sheet 8 | 2035 | |
| `sanborn08912_005_0005` | Burlington VT, 1906, sheet 5 | 1126 | |
| `sanborn09064_005_0003` | Richmond VA, 1919, sheet 3 | 702 | |

More regions means more merging, not a harder sheet. Before you start, read the notes for your sheet in
[Notes per sheet](#notes-per-sheet).

Every sheet is labeled **whole** (all buildings on it). A sheet takes a few hours; you can stop at any
time and continue later, nothing is lost.

### Notes per sheet

These are the places where the program is known to get it wrong on each sheet. Positions are given on the sheet
as you see it with **F** (whole sheet on screen).

**Denver 1903 (`sanborn00985_005_0006`), the test sheet.** It is labeled whole like the others, but it was
deliberately not tuned, so expect more missed walls to cut (**C**) than on the other sheets. Teal patches inside
pink buildings are part of the building: merge them in.

**San Francisco 1899, sheet 6 (`sanborn00813_008_0006`).** Dark, stained paper. Near the bottom left, one region runs
from the Sal. & Lunch building into the D./Restt. buildings next to it: cut it. Dashed lines between stores are often
not cut (the stores along the top right and top left blocks): cut them where they separate buildings. Drop the sheet
number (top right), a strip near the top left, the scale bar (bottom) and the binding pieces along the left edge. The
dashed frame building by the wharf sheds at the bottom right has no region: put an ignore box (**I**) on it.

**San Francisco 1899, sheet 7 (`sanborn00813_008_0007`).** Drop the big stain region inside the Palace Hotel outline
(bottom left), the margin strip along the right edge, the binding pieces along the left edge and the fire-alarm
circles. In the Popular Restaurant building (top left) the solid wall between Nos. 35 and 33 is not cut: cut it. The
dashed rows of the offices in the bottom right block (Nos. 628-640) need cuts where they separate buildings.
Slanted walls of the triangular blocks are cut correctly.

**Savannah 1898, sheet 12 (`sanborn01500_003_0012`).** In the S.P. Shotter plant (inset at the top right), the
rotated Storage / tank room / Cooper Shop chain is one region: about 4 cuts along its slanted partitions. The gray oil
tanks there have no color and no region: ignore boxes. In Alfred Kent's Varnish Room / Wood Working building, the
L-shaped dashed partition needs a cut. Drop the ~20 small junk regions along the water-pipe lines on Pine St., the
scale bar, the Library of Congress stamp and the margins. X-hatched sheds come in triangles: merge them.

**Sacramento 1895, sheet 10 left half (`sanborn00799_001_0010L`).** This is the left half of a two-page spread and very
stained. Buildings touching the right edge of the image have no region (the Adams Bldg in the upper right block,
almost all of the small block at the bottom right): ignore boxes. Yellow frame strips with dashed outer sides around
the blocks are often missing too (all around the Western Hotel block, bottom right; the "Shed (old) on posts" column
in the top left block): ignore boxes. Stained pink buildings (Western Hotel rooms, Sacramento Cracker Co.) come in many
ragged pieces: merge them. Drop the stained letters of street names ("SECOND", "80 feet wide") and the cistern circle
near the bottom (it is in 2 pieces). Dashed partitions are cut now; merge them back where they are inside one
building.

**Salt Lake City 1898, sheet 8 (`sanborn08891_003_0008`).** The most regions per building of all sheets: buildings full
of text (the Natatorium, the Undertaker shop) come in many small pieces, and pink brick bands are split off yellow,
green and olive buildings. Merge them back (**Shift+drag** is fastest). Olive adobe dwellings ("D.") joined to a
yellow shed by only a dashed line stay one region: cut them. Also cut the wall stubs in the yellow Carriage Shed
(middle). Drop the junk along the bottom left margin and the address strip along E. Temple St. (right edge).
X-hatched boxes come in pieces: merge them.

**Burlington 1906, sheet 5 (`sanborn08912_005_0005`).** About 75 tiny regions are just the inside of a written "D",
"2" or "o": merge each into its house. X-marked sheds come in 4-8 triangles and porches with story numbers in
several pieces, so expect 2-4 regions per house. Drop the small yellow blobs that bled outside houses. The house at the
very top edge (right half) has no region: ignore box. The corrugated-iron warehouse ("Ware Ho.", bottom left) only
covers its yellow inside; that is fine.

**Richmond 1919, sheet 3 (`sanborn09064_005_0003`).** X-hatched sheds come in triangles: merge them. Small paper pieces
next to dashed porches and the insides of letters ("D.", POST OFFICE) appear as regions: merge or drop. Partitions
that end inside a room may need a cut. The dashed yellow porch under the blue building (bottom middle) and outside
stairs have no region: ignore boxes.

---

## One-time setup

You do this once per computer. It works on Windows, Mac and Linux. Allow about 20 minutes.

### Step 1. Install GitHub Desktop (to download and upload the project)

1. Make a free account at <https://github.com/signup> if you don't have one, and send your GitHub
   username to the project lead so they can give you access to the project.
2. Download and install **GitHub Desktop** from <https://desktop.github.com>. Open it and sign in
   with your GitHub account.

(If you already use Git on the command line, you can use that instead; the commands are in
[Sending your work](#sending-your-work-git).)

### Step 2. Get the project onto your computer (clone)

1. Once the project lead has added you, in GitHub Desktop choose **File → Clone repository**.
2. Pick the project (`sanborn_labeling`) from the list, or paste the link the lead sent you in the **URL** tab.
3. Choose where to put it under **Local path** (for example your Documents folder) and click **Clone**.

You now have a folder called `sanborn_labeling` on your computer. This README is in it.

### Step 3. Put the sheet images in place

The map images (`.jp2` files, about 10 MB each) are too big for Git, so the project lead shares them
separately (for example a Google Drive folder or a zip file).

1. Download the `.jp2` file of your sheet (or all 8, that's fine too). If it is a zip, unzip it.
2. Move the `.jp2` files into the folder **`sanborn_labeling/data/raw/`**. Don't rename them.
   Afterwards that folder holds files like `data/raw/sanborn01500_003_0012.jp2`.

### Step 4. Install uv (it runs the labeling program)

uv installs Python and everything else the program needs, by itself. Open a terminal (see
[Opening a terminal](#opening-a-terminal-in-the-project-folder); any folder is fine for this step) and paste the line for your system:

- **Mac or Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows** (in PowerShell):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Then **close the terminal and open a new one** (so it finds uv). Check it worked: `uv --version` should print a
version number.

### Step 5. Install the program and check your sheets

Open a terminal **in the project folder** ([how](#opening-a-terminal-in-the-project-folder)) and run:

```bash
uv sync
```

The first time it downloads Python and a few libraries (about 150 MB, a minute or two). Then run:

```bash
uv run sheets
```

It lists the 8 sheets. Your sheet should say `ok` in the **image** column. `MISSING` means its `.jp2` is not in
`data/raw/` yet (Step 3). Setup is done.

---

## Opening a terminal in the project folder

Every command in this README is typed into a terminal that is "in" the `sanborn_labeling` folder.

- **Windows:** open the `sanborn_labeling` folder in File Explorer, click the address bar at the top, type
  `powershell` and press Enter. (On Windows 11 you can also right-click inside the folder → **Open in Terminal**.)
- **Mac:** open the **Terminal** app, type `cd ` (with a space after it), drag the `sanborn_labeling` folder from
  Finder into the Terminal window, and press Enter.
- **GitHub Desktop** can do it for you: **Repository → Open in Command Prompt / Terminal**.

To check you are in the right place, run `uv run sheets`: it prints the sheet table. If it says something
like "No `project` table found", you are in the wrong folder.

To paste into a terminal: **Ctrl+V** (Windows; or right-click), **Cmd+V** (Mac).

---

## Labeling a sheet

### Start the labeling tool

In a terminal in the project folder, run (put your sheet id in place of the example):

```bash
uv run label sanborn01500_003_0012
```

The first time, it takes up to a minute to prepare the image. Then your web browser opens
**http://127.0.0.1:8765** with the map. (If it doesn't open by itself, open that address in Chrome,
Firefox, Edge or Safari.) This page runs only on your computer, not on the internet.

**Keep the terminal window open while you label.** Every change is saved the moment you make it, so when you
are done for the day just close the browser tab and press **Ctrl+C** in the terminal (or close the terminal).
To continue later, run the same `uv run label ...` command again: everything is where you left it.

Open only **one** browser tab of the tool at a time. A second tab is refused so it can't overwrite your work.

### What you see

- The **map**, with every region outlined. Drag to move around, use the mouse wheel (or pinch on a touchpad)
  to zoom, press **F** to see the whole sheet.
- The **panel** on the right lists the steps for this sheet and ticks them off, shows the selected building
  and the keys that make sense right now, and has the full key list. **Tab** hides/shows the panel.
- Colors of the outlines (also explained at the bottom of the panel): not checked yet, checked (green),
  dropped, flagged (orange: probably not a building), suspect (dashed pink: may hold two buildings),
  warning (dashed red).
- Press **T** to switch the view between states / colors / plain map, and hold **H** to hide all outlines
  for a moment and look at the map underneath.

### What counts as one building (read this before you start)

A **building** is a structure with its own walls. On these maps:

- An attached part (porch, wing, addition, shed) is **its own building** when it is closed off by a
  **full-weight (thick) wall** AND has **its own color or its own story number** (the small 1, 2, 3 written in it).
- It **belongs to the main building** when it is only separated by a **thin line** inside the main outline.
- Buildings in a row that share a wall (row houses, stores) are **separate buildings**, one per wall-to-wall unit.
- Galleries and awnings (dashed outlines along a sidewalk) are their own buildings.
- **Not buildings, drop them:** fire-alarm boxes and numbered circles, water tanks and cisterns, street
  names, numbers and other text, symbols, stains, the paper edge and margins.
- A real structure the program missed entirely (no region): draw an **ignore box** around it (**I**) so it
  doesn't count as background.

The panel shows this rule too. If you meet a case you are unsure about, write it in
`docs/attached_part_rule.md` (sheet, where, what you decided) and ask the project lead, so everyone decides
the same way.

### Step 1: Use the whole sheet

The first time you open a sheet, click **Use whole sheet** in the panel. (The Denver sheet is a test sheet and
is always whole, so this step is already ticked there.)

### Step 2: Review queue (press N)

Press **N**. The tool zooms to the next item it is unsure about, with a card saying what it is and what to press:

- **Flagged** (orange) regions are almost certainly not buildings (circles, tanks, margin marks, text on yellowed
  paper). **D** drops it, **K** keeps it.
- **Suspect** (dashed pink) regions have a straight wall running inside them and may hold two buildings.
  If there really is a wall: press **C**, click along the wall from one side of the region fully across to the
  other side, and press **Enter**. If not: **K** keeps it.
- **Warnings** (dashed red) are buildings whose regions don't touch. Look and fix the merge.

Press **N** again for the next item until the panel says **Review queue done**.

### Step 3: Merge and check every building (Space)

This is most of the work. Repeat until the panel says **Every region checked**:

1. Press **Space**. The tool selects the nearest region that is not checked yet.
2. Look at the map: which building does this region belong to?
   - The building has more regions: **Shift+click** each of them to add it (Shift+click a region again to take it
     out). **Shift+drag** across several regions adds every region the stroke passes over.
   - The region is not a building: press **D** to drop it.
   - A wall is missing and the region holds two buildings: **C**, click across along the wall, **Enter**.
3. When the selected building is right, press **Space** again. It turns green (checked) and the next region is
   selected.

A region you leave alone is a building on its own once checked. To redo a building, select it and press
**Delete** (or the panel's Reset button); its regions go back to separate and unchecked. **Ctrl+Z** (Cmd+Z on a Mac)
undoes your last change.

Useful keys while merging:

| Key | What it does |
|---|---|
| click | select a region and its building |
| Shift+click | add a touching region to the selected building, or take one of its regions out |
| Shift+drag | add every touching region the stroke passes over |
| Shift+Alt+click | join two whole buildings |
| Space | this building is done → go to the next unchecked region |
| D | drop / restore the selected building (Alt+click: drop / restore one region) |
| Delete | reset the selected building |
| C | cut a missed wall (click points across the region, Enter to finish, Esc to cancel) |
| I | ignore box around a structure that has no region |
| B | mark everything on screen checked (only when you've looked at all of it!) |
| / | go to a region by its number |
| F | fit the whole sheet on screen |
| T | change the view · hold H to hide outlines |
| Ctrl+Z / Cmd+Z | undo |

Don't worry about outlines that wobble a little, spill a few pixels past a dashed porch line, or are notched
where a number touches a wall: the dissolve step moves every outline onto its drawn walls by itself. You only
decide **which regions form which building**.

### Tips

- Work block by block, zoomed in far enough to see the walls clearly. Thick black lines are walls between
  buildings; thin lines inside a building are rooms or details.
- Look at the colors: each building usually has one color (pink = brick, yellow = wood frame, blue = stone,
  green, gray, ...). A change of color is usually a change of building.
- Take breaks and push your work at the end of every session (see [Sending your work](#sending-your-work-git)).
  That is also your backup.
- If the page shows a **red banner** saying the server stopped, the terminal was closed: run `uv run label ...`
  again, the page reconnects by itself.

---

## Dissolve and check the result

When the panel shows all steps ticked, it shows the dissolve command with a **copy** button. In a terminal in the
project folder (stop the labeling tool first with Ctrl+C, or open a second terminal), run:

```bash
uv run dissolve sanborn01500_003_0012
```

It takes about a minute and prints how many buildings it found. It writes the final labels into
`data/labels/<your sheet>/`. Then **open `qa.jpg` in that folder** (double-click it): every building is shown in its
own color. Look over it for buildings merged together or a building cut in two. If you see a mistake, open the
labeling tool again, fix it, and run dissolve again.

The dissolve may print warnings, for example "3 buildings are in several pieces (ids ...)": open the tool, press
**/** and type that id to go there, and fix the merge.

Optional, after a dissolve: in the tool, **O** shows the final outlines. If one is wrong somewhere (spilling past a
wall), select the building and press **W** to draw the wall where it should be; then run dissolve again.

`uv run sheets` shows the state of every sheet (not started / in progress / dissolved).

---

## Sending your work (Git)

Your work is the folder `data/labels/<your sheet>/`. Push it at the end of every session, even when the sheet is not
finished yet, and once more after the final dissolve.

### With GitHub Desktop

1. Stop the labeling tool first (Ctrl+C in its terminal).
2. In GitHub Desktop, click **Fetch origin** (top bar), and **Pull origin** if it offers that. This downloads your
   teammates' work.
3. On the left you see your changed files (all inside `data/labels/<your sheet>/`). If you see changed files
   outside your sheet's folder that you didn't mean to change, right-click them → **Discard changes**.
4. Bottom left, write a short summary, for example `Savannah 12: 200 buildings checked`, and click **Commit to main**.
5. Click **Push origin** (top bar). Done.

If pushing fails with "rejected" or "newer commits", click **Pull origin** and then **Push origin** again.

### With the command line

```bash
git pull
git add data/labels/sanborn01500_003_0012
git commit -m "Savannah 12: 200 buildings checked"
git push
```

(Use your own sheet id. If `git push` is rejected, run `git pull` and then `git push` again.)

---

## Problems and fixes

**`uv: command not found` / `'uv' is not recognized`**: close the terminal and open a new one after installing uv. If
it still fails, run the install line from Step 4 again.

**"No `project` table found" or "Failed to spawn: label"**: the terminal is not in the project folder. See
[Opening a terminal](#opening-a-terminal-in-the-project-folder).

**"The sheet image is missing: put ... .jp2 into the data/raw folder"**: do Step 3. The file name must stay exactly as
it was shared.

**"... is not a sheet of this project"**: check the sheet id (copy it from the table above or from `uv run sheets`).

**"Port 8765 is in use"**: the tool is already open in another terminal window. Use that one, or close it. To run two
sheets at once, add `--port 8766` to the second one and open http://127.0.0.1:8766.

**"Another labeling tool (process ...) is already open on ..."**: close the other one. If you are sure none is open
(for example after a crash or restart), delete the file `data/labels/<sheet>/server.lock` and start again.

**The page says the tab is out of date / refuses to save**: you have the tool open in two tabs, or you started the
tool again. Close all its tabs and reload http://127.0.0.1:8765 once.

**The first start is slow**: the first time a sheet opens, its image is decoded and stored in `data/cache/`
(about 150 MB per sheet). After that it opens in seconds. You can delete `data/cache/` at any time to free space.

**The computer gets slow or the program says out of memory**: close other programs. The tool needs about 2-3 GB
of free memory, dissolve a bit more.

**I made a mess of a building**: select it, press **Delete**, merge it again. Ctrl+Z undoes single steps.

**I want to start a sheet over completely**: ask the project lead. (It means deleting the sheet's `edits.json` and
`cuts.json`, which loses all work on it.)

**GitHub Desktop shows a conflict**: you and someone else changed the same file. Don't guess: ask the project lead.

---

## For the project lead

The regions in `data/labels/<sheet>/regions.png` and `regions.geojson` were made in advance with the heuristic
over-segmenter (`sanborn/oversegment.py`) from the main project, with the knobs in `config.yaml`: the shared
`oversegment:` values plus the city's entry in `oversegment_by_city:` (Denver, the test city, uses the shared values
only). They are never changed by the tool; cuts are stored separately in `cuts.json` and replayed. The team does not
run the over-segmenter. To add a sheet: run the over-segmenter on it in the main project, copy its
`outputs/overseg/<sheet>/labels.png` to `data/labels/<sheet>/regions.png` and `regions.geojson` next to it, add its row
to `data/manifest.csv` (the city decides test vs. training: `labels.test_cities` in `config.yaml`), share its `.jp2`,
and commit. How each city's knobs were chosen (label-free checks on the sheet itself, with a hold-out part)
is written up in `docs/tuning/<city>.md`.

What each labeled sheet folder holds: `regions.png` / `regions.geojson` (the pre-cut regions), `edits.json` (merges,
drops, checks, extents, ignore boxes, drawn walls, active minutes), `cuts.json` (drawn cuts), `folder.json` (an id that
stops stale tabs from saving), and after dissolve `instances.png` (uint16 instance map, 0 = background, 65535 =
ignore), `valid.png` (labeled area), `buildings.geojson`, `qa.jpg`, `summary.json` (buildings, party walls, split,
minutes) and `fitted.png` / `fitted.json` (for the tool's final view). Unlike the main project, dissolve here does not
write `data/manifest.csv`, so pushes from different labelers never conflict; the counts are in each `summary.json`.

Code: `sanborn/labeltool.py` + `labeltool.html` (the tool), `sanborn/labels.py` (dissolve and the wall fit),
`sanborn/oversegment.py` (over-segmenter; dissolve's wall fit uses its ink masks), `sanborn/status.py`
(`uv run sheets`). `uv run pytest` runs the tests. Differences from the main project's copy: the lock check works on
Windows (`os.kill(pid, 0)` terminates the process there), friendlier error messages, the browser opens by itself, the
`label` / `dissolve` / `sheets` shortcuts, the panel steers training sheets to "Use whole sheet", and dissolve doesn't
touch the manifest.
