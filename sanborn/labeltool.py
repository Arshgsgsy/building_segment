"""Browser labeling tool: merge pre-cut regions into buildings, drop non-buildings and cut missed walls.

Run `uv run label <sheet>` and open http://127.0.0.1:8765; the keys are listed in the page's side panel.
Every change is saved to data/labels/<sheet>/ right away; `uv run dissolve <sheet>` makes the labels.
"""
import argparse
import atexit
import json
import os
import shutil
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from sanborn import ROOT, load_config, manifest
from sanborn.data import load_sheet
from sanborn.labels import LABELS, apply_walls, cut, load_regions, neighbors, polygonize, uncut

TILE = 512
PAGE = Path(__file__).with_name("labeltool.html")


class Sheet:
    """One sheet's image pyramid, working region map, cuts and edits."""

    def __init__(self, name):
        self.name, self.dir = name, LABELS / name
        src = ROOT / "outputs" / "overseg" / name
        jp2 = ROOT / "data" / "raw" / f"{name}.jp2"
        if not (self.dir / "regions.png").exists():
            if not (src / "labels.png").exists():
                raise SystemExit(f"{name} is not a sheet of this project (no data/labels/{name}/regions.png).\n"
                                 "Check the spelling; `uv run sheets` lists the sheets.")
            self.dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / "regions.geojson", self.dir / "regions.geojson")
            shutil.copy(src / "labels.png", self.dir / "regions.png")
        if not jp2.exists():
            raise SystemExit(f"The sheet image is missing: put {name}.jp2 into the data/raw folder\n"
                             f"(full path: {jp2}). The .jp2 files are shared separately; see README.md, step 3.")
        # a new id whenever the folder is (re)created: a page left open from before is refused
        self.epoch = self.read("folder.json", {}).get("epoch")
        if not self.epoch:
            self.epoch = uuid.uuid4().hex
            self.write("folder.json", {"epoch": self.epoch})
        self.lock_folder()
        cfg = load_config()["labels"]
        self.min_piece, self.reach = cfg["min_piece"], cfg["merge_reach"]
        self.wall_width, self.wall_snap = cfg["wall_width"], cfg["wall_snap"]
        self.base, self.base_members, self.base_mtime = None, {}, None  # dissolve's buildings before the walls
        self.regions, self.cuts = load_regions(self.dir, self.min_piece)
        self.next_id = self.read("cuts.json", {}).get("next_id", 0)  # ids are never handed out twice
        feats = json.loads((self.dir / "regions.geojson").read_text())["features"]
        self.meta = {f["properties"]["id"]: f["properties"] for f in feats}
        self.parent = {int(k): v for c in self.cuts for k, v in c["pieces"].items()}
        self.lock = threading.Lock()

        # training sheets are labeled inside extents, the test city as a whole sheet
        row = next((r for r in manifest.read() if r["sheet_id"] == name), {})
        self.city = row.get("city", "")
        self.split = "test" if self.city.lower() in cfg["test_cities"] else "train"

        # image pyramid: level 0 is the sheet, each next level half the size, down to one tile
        self.levels = [load_sheet(jp2)]
        while max(self.levels[-1].shape[:2]) > TILE:
            h, w = self.levels[-1].shape[:2]
            self.levels.append(cv2.resize(np.asarray(self.levels[-1]), ((w + 1) // 2, (h + 1) // 2),
                                          interpolation=cv2.INTER_AREA))
        self.tiles = {}

    def read(self, name, default):
        path = self.dir / name
        return json.loads(path.read_text()) if path.exists() else default

    def write(self, name, obj):
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(json.dumps(obj))
        tmp.replace(self.dir / name)

    def lock_folder(self):
        """Refuse to start when another server is open on this sheet: both would write its files."""
        lock = self.dir / "server.lock"
        if lock.exists():
            try:
                pid = int(lock.read_text() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and pid_alive(pid):
                raise SystemExit(f"Another labeling tool (process {pid}) is already open on {self.name}.\n"
                                 "Stop it first (Ctrl+C in its terminal window), or delete "
                                 f"data/labels/{self.name}/server.lock if you are sure none is running.")
            # otherwise it is left over from a tool that crashed or a reboot
        lock.write_text(str(os.getpid()))
        atexit.register(lambda: lock.unlink(missing_ok=True))

    def current(self, body):
        """Whether a request comes from a page of this sheet and this label folder."""
        return body.get("sheet") == self.name and body.get("epoch") == self.epoch

    def save_edits(self, edits):
        """Save the page's edits if they build on the saved version; returns the new version, or None."""
        with self.lock:
            saved = self.read("edits.json", {}).get("rev", 0)
            if not self.current(edits) or edits.get("rev") != saved:
                return None
            edits = {k: v for k, v in edits.items() if k not in ("sheet", "epoch")}
            edits["rev"] = saved + 1
            self.write("edits.json", edits)
            return edits["rev"]

    def state(self):
        h, w = self.regions.shape
        return {"sheet": self.name, "epoch": self.epoch, "city": self.city, "split": self.split, "width": w, "height": h,
                "tile": TILE, "levels": len(self.levels), "regions": self.outlines(),
                "adjacent": neighbors(self.regions, self.reach), "edits": self.read("edits.json", {}),
                "cuts": len(self.cuts)}

    def outlines(self, ids=None):
        """Region outlines for the page: all rings (outer and holes) in sheet pixels, plus flags."""
        if ids is None:
            polys = polygonize(self.regions)
        else:
            polys = {}
            for i in ids:
                ys, xs = np.nonzero(self.regions == i)
                crop = self.regions[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                polys.update(polygonize(np.where(crop == i, i, 0).astype(np.int32), offset=(xs.min(), ys.min())))
        out = []
        for i, g in polys.items():
            m = self.meta.get(i, {})
            out.append({"id": i, "rings": rings(g), "flag": m.get("flag", ""), "suspect": m.get("suspect", 0),
                        "parent": self.parent.get(i)})
        return out

    def load_base(self):
        """(Re)load fitted.png / fitted.json when a dissolve wrote new ones; False if there is none yet."""
        path = self.dir / "fitted.png"
        if not path.exists() or not (self.dir / "fitted.json").exists():
            self.base, self.base_mtime = None, None
            return False
        mtime = path.stat().st_mtime
        if mtime != self.base_mtime:
            self.base = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.base_members = {int(b): rs for b, rs in self.read("fitted.json", {})["members"].items()}
            self.base_mtime = mtime
        return True

    def final(self, body):
        """The last dissolve's buildings with the page's walls applied, and the pixels each wall changed."""
        with self.lock:
            if not self.current(body):
                return 409, {"error": "stale"}
            if not self.load_base():
                return 200, {"available": False}
            base, members, mtime = self.base, self.base_members, self.base_mtime
        ids = None if body.get("ids") is None else {int(i) for i in body["ids"]}
        work = base.copy()
        done = apply_walls(work, body.get("walls", []), members, self.wall_width, self.wall_snap, only=ids)
        polys = {}
        if ids is None:
            polys = polygonize(work)
        else:
            for b in ids:
                ys, xs = np.nonzero(work == b)
                if len(ys):
                    crop = work[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    polys.update(polygonize(np.where(crop == b, b, 0).astype(np.int32), offset=(xs.min(), ys.min())))
        return 200, {"available": True, "dissolved": mtime, "results": done,
                     "buildings": [{"id": b, "regions": members.get(b, []), "rings": rings(g)} for b, g in polys.items()],
                     "gone": sorted((ids or set(members)) - set(polys))}

    def tile(self, z, x, y):
        if (z, x, y) not in self.tiles:
            t = np.ascontiguousarray(self.levels[z][y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE])
            if not t.size:
                return None
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(t, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.tiles[z, x, y] = buf.tobytes()
        return self.tiles[z, x, y]

    def cut(self, body):
        """Apply a drawn cut, refusing a page that missed an earlier one; returns (status, reply)."""
        with self.lock:
            if not self.current(body) or body.get("cuts") != len(self.cuts):
                return 409, {"error": "stale"}
            next_id = max(int(self.regions.max()) + 1, max(self.meta, default=0) + 1, self.next_id)
            try:
                pieces = cut(self.regions, body["line"], next_id, self.min_piece)
            except ValueError as e:
                return 400, {"error": str(e)}
            if not pieces:
                return 400, {"error": "The line must run fully across a region."}
            self.cuts.append({"line": body["line"], "min_piece": self.min_piece,
                              "pieces": {str(k): v for k, v in pieces.items()}})
            self.parent.update(pieces)
            self.next_id = max(pieces) + 1
            self.save_cuts()
            return 200, self.changed(sorted(set(pieces.values())), list(pieces))

    def uncut(self, body):
        """Undo the last cut (cuts are undone newest first, so nested pieces come back in order)."""
        with self.lock:
            if not self.current(body) or body.get("cuts") != len(self.cuts):
                return 409, {"error": "stale"}
            if not self.cuts:
                return 400, {"error": "No cut to undo."}
            pieces = {int(k): v for k, v in self.cuts.pop()["pieces"].items()}
            uncut(self.regions, pieces)
            for k in pieces:
                self.parent.pop(k, None)
            self.save_cuts()
            return 200, self.changed(list(pieces), sorted(set(pieces.values())))

    def save_cuts(self):
        self.write("cuts.json", {"cuts": self.cuts, "next_id": self.next_id})

    def changed(self, removed, added):
        """What the page needs after a cut or uncut: removed ids, new outlines and their neighbors."""
        return {"removed": removed, "added": self.outlines(added),
                "adjacent": neighbors(self.regions, self.reach, set(added))}


def pid_alive(pid):
    """Whether a process with this id is running. Works on Windows, where os.kill(pid, 0) would kill it."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.GetLastError() == 5  # access denied: it exists but belongs to someone else
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def rings(g):
    """All rings (outer and holes) of a polygon or multipolygon, as integer sheet pixels."""
    return [np.round(r.coords).astype(int).tolist() for p in getattr(g, "geoms", [g]) for r in (p.exterior, *p.interiors)]


def handler(sheet):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, ctype="application/json", code=200, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code=200):
            self.send(json.dumps(obj).encode(), code=code)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self.send_json(sheet.state())
            if path.startswith("/tile/"):
                z, x, y = map(int, path[6:].removesuffix(".jpg").split("/"))
                if 0 <= z < len(sheet.levels) and x >= 0 and y >= 0 and (body := sheet.tile(z, x, y)):
                    return self.send(body, "image/jpeg", cache=True)
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/edits":
                    rev = sheet.save_edits(body)
                    return self.send_json({"rev": rev} if rev else {"error": "stale"}, 200 if rev else 409)
                if self.path == "/api/cut":
                    code, reply = sheet.cut(body)
                    return self.send_json(reply, code)
                if self.path == "/api/uncut":
                    code, reply = sheet.uncut(body)
                    return self.send_json(reply, code)
                if self.path == "/api/final":
                    code, reply = sheet.final(body)
                    return self.send_json(reply, code)
                self.send(b"not found", "text/plain", 404)
            except Exception as e:  # tell the page, so it doesn't mistake an error for a stopped server
                print(f"error on {self.path}: {e!r}")
                self.send_json({"error": repr(e)}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", help="sheet id, e.g. sanborn07492_002_0003 (a path to the .jp2 works too)")
    ap.add_argument("--port", type=int, default=8765, help="use another port (e.g. 8766) if 8765 is taken")
    ap.add_argument("--no-browser", action="store_true", help="don't open the page in the browser automatically")
    a = ap.parse_args()

    print(f"loading {Path(a.sheet).stem} (the first time takes up to a minute) ...")
    sheet = Sheet(Path(a.sheet).stem)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(sheet))
    except OSError:
        raise SystemExit(f"Port {a.port} is in use (another labeling tool is probably open in another window).\n"
                         f"Close that one, or start this one on another port: uv run label {sheet.name} --port {a.port + 1}")
    url = f"http://127.0.0.1:{a.port}"
    print(f"{sheet.name} ({sheet.city}): {len(sheet.meta)} regions, {len(sheet.cuts)} cuts so far")
    print(f"labeling tool at {url}")
    print("Keep this window open while you label. Every change is saved at once.")
    print("To stop: press Ctrl+C here (or just close this window).")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped; your edits are saved.")


if __name__ == "__main__":
    main()
