"""Build viz/brain.js, the data behind the interactive page in viz/index.html.

Inputs: the FlyWire v783 annotations and connectivity in data/raw (see
flyneuromod.data.download), the FlyWire whole-brain outline, and the JFRC2
neuropil meshes (Ito et al. 2014) in FlyWire space as shipped by fafbseg-py.
The two mesh sources are downloaded here and pinned by URL.

Usage::

    uv run --with trimesh python viz/export_data.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
import pathlib
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import trimesh

logger = logging.getLogger(__name__)

OUT = pathlib.Path(__file__).resolve().parent
CACHE = OUT / ".cache"
RAW = pathlib.Path("data/raw")
BRAIN_SURF_URL = (
    "https://storage.googleapis.com/flywire_neuropil_meshes/"
    "whole_neuropil/brain_mesh_v141.surf/mesh/1:0:0"
)
NEUROPILS_URL = (
    "https://raw.githubusercontent.com/navis-org/fafbseg-py/master/"
    "fafbseg/data/JFRC2NP.surf.fw.zip"
)
NEUROPILS = [
    f"{region}_{side}"
    for region in ("MB_CA", "MB_PED", "MB_VL", "MB_ML", "AL", "LH")
    for side in ("L", "R")
]
VOXEL_UM = (0.004, 0.004, 0.040)  # FlyWire voxel size in micrometres
ODOUR_FRACTION = 0.10  # share of Kenyon cells the page's odour drives
FLAG_ODOUR, FLAG_WIRED = 1, 2
COLUMNS = ["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z",
           "super_class", "cell_class", "cell_type"]


def fetch(url: str, name: str) -> pathlib.Path:
    """Download ``url`` into the cache once."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def categories(a: pd.DataFrame) -> np.ndarray:
    """Draw category of every neuron; later rules win."""
    cat = np.zeros(len(a), np.uint8)
    cat[a.super_class.isin(["optic", "visual_projection", "visual_centrifugal"])] = 1
    cat[a.cell_class == "Kenyon_Cell"] = 2
    cat[a.cell_class == "DAN"] = 3
    cat[a.cell_class == "MBON"] = 4
    cat[a.cell_type.isin(["PPL101", "MBON11"])] = 5
    cat[a.cell_type.isin(["PAM01", "PAM15"])] = 6
    cat[a.cell_type == "MBON01"] = 7
    return cat


def load_neuropils(path: pathlib.Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named neuropil meshes, vertices in micrometres."""
    meshes = {}
    with zipfile.ZipFile(path) as archive:
        for name in NEUROPILS:
            mesh = trimesh.load(io.BytesIO(archive.read(f"{name}.ply")), file_type="ply")
            meshes[name] = (np.asarray(mesh.vertices) / 1000, np.asarray(mesh.faces))
    return meshes


def load_brain_outline(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Neuroglancer legacy mesh fragment: vertex count, float32 xyz, uint32 faces."""
    raw = path.read_bytes()
    n_vertices = int(np.frombuffer(raw[:4], "<u4")[0])
    end = 4 + 12 * n_vertices
    vertices = np.frombuffer(raw[4:end], "<f4").reshape(-1, 3).astype(np.float64) / 1000
    faces = np.frombuffer(raw[end:], "<u4").reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.merge_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def midline(meshes: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """x of the midline, from paired neuropils (the point cloud is asymmetric)."""
    pairs = ["MB_CA", "MB_PED", "AL", "LH"]
    return float(np.mean([
        (meshes[f"{p}_L"][0].mean(0)[0] + meshes[f"{p}_R"][0].mean(0)[0]) / 2 for p in pairs
    ]))


def wiring_flags(a: pd.DataFrame, cat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Mark the odour-driven Kenyon cells and those wired to MBON01 (MBON-gamma5)."""
    connections = pd.read_parquet(
        RAW / "Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID"]
    )
    mbon_ids = a.loc[a.cell_type == "MBON01", "root_id"].tolist()
    edges = connections[connections.Postsynaptic_ID.isin(mbon_ids)]
    index_of = pd.Series(a.index.values, index=a.root_id)
    flags = np.zeros(len(a), np.uint8)
    rng = np.random.default_rng(7)
    flags[(cat == 2) & (rng.random(len(a)) < ODOUR_FRACTION)] |= FLAG_ODOUR
    pre = index_of.reindex(edges.Presynaptic_ID).dropna().astype(int).to_numpy()
    flags[np.unique(pre[cat[pre] == 2])] |= FLAG_WIRED
    return flags, mbon_ids


def anchors(meshes, mid: float, soma: np.ndarray, a: pd.DataFrame, cat, mbon_ids) -> dict:
    """Landmarks the page flies to, per hemisphere."""
    def gamma5(v: np.ndarray) -> np.ndarray:
        # the medial tip of the medial lobe, anterior half: an estimate of gamma5
        d = np.abs(v[:, 0] - mid)
        medial = v[d <= np.quantile(d, 0.2)]
        return medial[medial[:, 2] <= np.median(medial[:, 2])].mean(0)

    def junction(v: np.ndarray) -> np.ndarray:
        d = np.abs(v[:, 0] - mid)
        return v[d >= np.quantile(d, 0.85)].mean(0)

    points = {}
    for side in ("L", "R"):
        medial_lobe = meshes[f"MB_ML_{side}"][0]
        points[f"g5{side}"] = gamma5(medial_lobe)
        points[f"jn{side}"] = junction(medial_lobe)
        for region, key in (("MB_PED", "ped"), ("AL", "al"), ("MB_CA", "ca"), ("LH", "lh")):
            points[f"{key}{side}"] = meshes[f"{region}_{side}"][0].mean(0)
    # each MBON01 belongs to the side most of its Kenyon cell inputs come from
    right = soma[:, 0] >= mid
    connections = pd.read_parquet(
        RAW / "Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID"]
    )
    index_of = pd.Series(a.index.values, index=a.root_id)
    for mbon in mbon_ids:
        inputs = connections.loc[connections.Postsynaptic_ID == mbon, "Presynaptic_ID"]
        pre = index_of.reindex(inputs).dropna().astype(int).to_numpy()
        pre = pre[cat[pre] == 2]
        side = "R" if right[pre].mean() > 0.5 else "L"
        points["mbon" + side] = soma[index_of[mbon]]
    points["mb"] = np.mean([points["g5L"], points["g5R"], points["caL"], points["caR"]], axis=0)
    return points


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    neuropil_zip = fetch(NEUROPILS_URL, "np.zip")
    outline = fetch(BRAIN_SURF_URL, "brain_surf.bin")
    a = pd.read_csv(RAW / "flywire_annotations_v783.tsv", sep="\t", usecols=COLUMNS,
                    low_memory=False)
    pos = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_UM
    soma = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_UM
    soma = np.where(np.isnan(soma[:, :1]), pos, soma)  # no soma recorded: use pos
    cat = categories(a)

    meshes = load_neuropils(neuropil_zip)
    mid = midline(meshes)
    meshes["BRAIN"] = load_brain_outline(outline)
    flags, mbon_ids = wiring_flags(a, cat)
    points = anchors(meshes, mid, soma, a, cat, mbon_ids)

    everything = np.vstack([soma, pos[~np.isnan(pos[:, 0])]] + [v for v, _ in meshes.values()])
    lo, hi = everything.min(0), everything.max(0)

    def quantize(x: np.ndarray) -> np.ndarray:
        return np.round((x - lo) / (hi - lo) * 65535).astype("<u2")

    arbor = np.where(np.linalg.norm(pos - soma, axis=1)[:, None] > 3, pos, soma)
    order = np.argsort(cat, kind="stable")
    values, counts = np.unique(cat, return_counts=True)
    meta = {
        "n": len(a), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist(), "mid": round(mid, 2),
        "counts": {int(k): int(v) for k, v in zip(values, counts, strict=True)},
        "anchors": {k: np.asarray(v).round(2).tolist() for k, v in points.items()},
        "wired": int(((flags & FLAG_WIRED) > 0).sum()), "meshes": {},
    }
    data = {"soma": b64(quantize(soma[order])), "arbor": b64(quantize(arbor[order])),
            "cat": b64(cat[order]), "flags": b64(flags[order])}
    for name, (vertices, faces) in meshes.items():
        data[f"mv_{name}"] = b64(quantize(vertices))
        data[f"mf_{name}"] = b64(faces.astype("<u2"))
        meta["meshes"][name] = {"nv": len(vertices), "nf": len(faces)}
    js = "window.BRAIN=" + json.dumps(meta) + ";\n"
    js += "".join(f"window.BRAIN.{k}='{v}';\n" for k, v in data.items())
    (OUT / "brain.js").write_text(js)
    logger.info("brain.js: %.2f MB, %d wired Kenyon cells", len(js) / 1e6, meta["wired"])


if __name__ == "__main__":
    main()
