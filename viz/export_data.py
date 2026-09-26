"""Build viz/brain.js, the data behind the interactive page in viz/index.html.

Inputs: the FlyWire v783 annotations and connectivity in data/raw (see
flyneuromod.data.download), the FlyWire whole-brain outline, and the JFRC2
neuropil meshes (Ito et al. 2014) in FlyWire space as shipped by fafbseg-py.
The two mesh sources are downloaded here and pinned by URL.

Usage::

    uv run --with trimesh python viz/export_data.py
"""
import sys, base64, json, io, zipfile, pathlib, urllib.request
import numpy as np, pandas as pd, trimesh

OUT = pathlib.Path(__file__).resolve().parent
CACHE = OUT / ".cache"
BRAIN_SURF_URL = "https://storage.googleapis.com/flywire_neuropil_meshes/whole_neuropil/brain_mesh_v141.surf/mesh/1:0:0"
NEUROPILS_URL = "https://raw.githubusercontent.com/navis-org/fafbseg-py/master/fafbseg/data/JFRC2NP.surf.fw.zip"

def fetch(url, name):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path

sp = str(CACHE)
fetch(NEUROPILS_URL, "np.zip"); fetch(BRAIN_SURF_URL, "brain_surf.bin")
b64 = lambda arr: base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()
a = pd.read_csv("data/raw/flywire_annotations_v783.tsv", sep="\t",
    usecols=["root_id","pos_x","pos_y","pos_z","soma_x","soma_y","soma_z","super_class","cell_class","cell_type"], low_memory=False)
pos = np.c_[a.pos_x*0.004, a.pos_y*0.004, a.pos_z*0.040]
som = np.c_[a.soma_x*0.004, a.soma_y*0.004, a.soma_z*0.040]
has = ~np.isnan(som[:,0]); S = np.where(has[:,None], som, pos)
cat = np.zeros(len(a), np.uint8)
cat[a.super_class.isin(["optic","visual_projection","visual_centrifugal"])] = 1
cat[a.cell_class == "Kenyon_Cell"] = 2
cat[a.cell_class == "DAN"] = 3
cat[a.cell_class == "MBON"] = 4
cat[a.cell_type.isin(["PPL101","MBON11"])] = 5
cat[a.cell_type.isin(["PAM01","PAM15"])] = 6
cat[a.cell_type == "MBON01"] = 7
z = zipfile.ZipFile(f"{sp}/np.zip")
def load_ply(name):
    m = trimesh.load(io.BytesIO(z.read(f"{name}.ply")), file_type="ply"); return np.asarray(m.vertices)/1000, np.asarray(m.faces)
names = ["MB_CA_L","MB_CA_R","MB_PED_L","MB_PED_R","MB_VL_L","MB_VL_R","MB_ML_L","MB_ML_R","AL_L","AL_R","LH_L","LH_R"]
meshes = {nm: load_ply(nm) for nm in names}
# midline from paired neuropils
MID = float(np.mean([(meshes[f"{p}_L"][0].mean(0)[0] + meshes[f"{p}_R"][0].mean(0)[0]) / 2 for p in ["MB_CA","MB_PED","AL","LH"]]))
right = S[:,0] >= MID
raw = open(f"{sp}/brain_surf.bin","rb").read(); nv = np.frombuffer(raw[:4],"<u4")[0]
bv = np.frombuffer(raw[4:4+12*nv],"<f4").reshape(-1,3).astype(np.float64)/1000; bf = np.frombuffer(raw[4+12*nv:],"<u4").reshape(-1,3)
bm = trimesh.Trimesh(bv, bf, process=True); bm.merge_vertices()
meshes["BRAIN"] = (np.asarray(bm.vertices), np.asarray(bm.faces))
# real KC -> MBON01 wiring; KCs connect ipsilaterally, so a KC's side is its own
c = pd.read_parquet("data/raw/Connectivity_783.parquet", columns=["Presynaptic_ID","Postsynaptic_ID"])
mb_ids = a.loc[a.cell_type=="MBON01","root_id"].tolist()
e = c[c.Postsynaptic_ID.isin(mb_ids)]
rid2idx = pd.Series(a.index.values, index=a.root_id)
pre_idx = rid2idx.reindex(e.Presynaptic_ID).values
flags = np.zeros(len(a), np.uint8)
rng = np.random.default_rng(7)
flags[(cat==2) & (rng.random(len(a)) < 0.10)] |= 1                  # the Kenyon cells this odour drives
kc_in = np.unique(pre_idx[~np.isnan(pre_idx)].astype(int)); kc_in = kc_in[cat[kc_in]==2]
flags[kc_in] |= 2                                                    # wired to MBON-gamma5
mbon_side = {}
for mid_ in mb_ids:
    pres = rid2idx.reindex(e.loc[e.Postsynaptic_ID==mid_,"Presynaptic_ID"]).dropna().astype(int).values
    pres = pres[cat[pres]==2]; mbon_side[mid_] = int(right[pres].mean() > 0.5)
print("MID", round(MID,1), "wired KCs L/R", int((flags[~right]&2>0).sum()), int((flags[right]&2>0).sum()),
      "odour&wired", int(((flags&3)==3).sum()), "mbon sides", list(mbon_side.values()))
def g5_point(v):
    d = np.abs(v[:,0]-MID); med = v[d <= np.quantile(d, 0.2)]
    return med[med[:,2] <= np.median(med[:,2])].mean(0)
def junction(v):
    d = np.abs(v[:,0]-MID); return v[d >= np.quantile(d, 0.85)].mean(0)
anch = {}
for s in ("L","R"):
    ml = meshes[f"MB_ML_{s}"][0]
    anch[f"g5{s}"] = g5_point(ml); anch[f"jn{s}"] = junction(ml)
    for p, key in (("MB_PED","ped"),("AL","al"),("MB_CA","ca"),("LH","lh")): anch[f"{key}{s}"] = meshes[f"{p}_{s}"][0].mean(0)
for mid_, sd in mbon_side.items():
    anch["mbon" + ("R" if sd else "L")] = S[rid2idx[mid_]]
anch["mb"] = np.mean([anch["g5L"], anch["g5R"], anch["caL"], anch["caR"]], axis=0)
print({k: np.round(v,0).tolist() for k, v in anch.items()})
allv = np.vstack([S, pos[~np.isnan(pos[:,0])]] + [v for v, f in meshes.values()])
lo, hi = allv.min(0), allv.max(0)
Q = lambda x: np.round((x-lo)/(hi-lo)*65535).astype("<u2")
A = np.where(np.linalg.norm(pos-S,axis=1)[:,None] > 3, pos, S)
order = np.argsort(cat, kind="stable")
meta = {"n": int(len(a)), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist(), "mid": round(MID,2),
        "counts": {int(k): int(v) for k, v in zip(*np.unique(cat, return_counts=True))},
        "anchors": {k: np.asarray(v).round(2).tolist() for k, v in anch.items()},
        "wired": int((flags&2>0).sum()), "meshes": {}}
data = {"soma": b64(Q(S[order])), "arbor": b64(Q(A[order])), "cat": b64(cat[order]), "flags": b64(flags[order])}
for k, (v, f) in meshes.items():
    data[f"mv_{k}"] = b64(Q(v)); data[f"mf_{k}"] = b64(f.astype("<u2"))
    meta["meshes"][k] = {"nv": int(len(v)), "nf": int(len(f))}
js = "window.BRAIN=" + json.dumps(meta) + ";\n" + "".join(f"window.BRAIN.{k}='{v}';\n" for k, v in data.items())
open(OUT / "brain.js", "w").write(js)
print("brain.js", round(len(js)/1e6,2), "MB")
