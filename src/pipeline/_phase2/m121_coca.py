"""M-121: COCA coronary calcium segmentation on gated cardiac CT."""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2, xml.etree.ElementTree as ET
import pydicom
from common import DATA_ROOT, write_task, COLORS, fit_square, window_ct, to_rgb, overlay_mask, pick_annotated_idx

PID="M-121"; TASK_NAME="coca_coronary_calcium_seg"; FPS=3
PROMPT=("This is a gated cardiac CT slice from the COCA dataset. "
        "Segment coronary artery calcium (red overlay) on each slice where calcium is visible.")

def process_case(dcm_dir: Path, xml_path: Path, idx: int):
    slices = sorted(dcm_dir.glob("*.dcm"))
    if not slices: return None
    # Read DICOM stack
    vols = []
    for s in slices:
        try:
            d = pydicom.dcmread(str(s))
            vols.append(d.pixel_array.astype(np.int16))
        except: continue
    if not vols: return None
    img_vol = np.stack(vols)
    # Parse XML calcium annotations (SIS format: ImageIndex → list of polygons)
    lbl_vol = np.zeros_like(img_vol, dtype=np.uint8)
    try:
        tree = ET.parse(xml_path).getroot()
        for img_node in tree.findall(".//Image"):
            idx_node = img_node.find("ImageIndex") or img_node.find("ImageNumber")
            if idx_node is None: continue
            slice_i = int(idx_node.text) - 1
            if slice_i < 0 or slice_i >= lbl_vol.shape[0]: continue
            for pt_list in img_node.findall(".//Pixels") or img_node.findall(".//Roi"):
                pts = []
                for p in pt_list.findall(".//Point") or pt_list.findall(".//pixel"):
                    try: pts.append([int(float(p.get("x"))), int(float(p.get("y")))])
                    except: pass
                if len(pts) >= 3:
                    cv2.fillPoly(lbl_vol[slice_i], [np.array(pts, dtype=np.int32)], 1)
    except: pass
    n=img_vol.shape[0]; step=max(1,n//60); idxs=list(range(0,n,step))[:60]
    ff,lf,gf,fl=[],[],[],[]
    for z in idxs:
        ct=window_ct(img_vol[z]); rgb=to_rgb(ct); rgb=fit_square(rgb,512)
        lab=fit_square(lbl_vol[z].astype(np.int16),512).astype(np.uint8)
        ann=overlay_mask(rgb,lab,color=COLORS["red"],alpha=0.55)
        ff.append(rgb); lf.append(ann); fl.append(bool(lab.sum()))
        if lab.sum(): gf.append(ann)
    if not gf: gf=lf[:5]
    pick=pick_annotated_idx(fl)
    meta={"task":"COCA coronary artery calcium segmentation","dataset":"COCA","case_id":dcm_dir.name,
          "modality":"gated cardiac CT","classes":["coronary_calcium"],"colors":{"coronary_calcium":"red"},
          "fps_source":"case B slice sequence fps=3","num_slices_total":int(n),"num_slices_used":len(idxs)}
    return write_task(PID,TASK_NAME,idx,ff[pick],lf[pick],ff,lf,gf,PROMPT,meta,FPS)

def main():
    root=DATA_ROOT/"_extracted"/"M-121_COCA"
    # Look for subject dirs (SIS schema: <subject>/<series>/ with .dcm + .xml)
    cases=[]
    for d in root.rglob("*"):
        if not d.is_dir(): continue
        dcms=list(d.glob("*.dcm"))
        if not dcms: continue
        xmls=list(d.parent.glob("*.xml"))+list(d.glob("*.xml"))
        if xmls: cases.append((d,xmls[0]))
    print(f"  {len(cases)} COCA cases")
    for i,(dcm,xml) in enumerate(cases):
        d=process_case(dcm,xml,i)
        if d: print(f"  wrote {d}")

if __name__=="__main__": main()
