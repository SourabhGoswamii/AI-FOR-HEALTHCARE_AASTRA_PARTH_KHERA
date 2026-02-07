# ============================================================
# 🧠 FULL MRI PREPROCESSING PIPELINE (FINAL PROFESSIONAL)
# From RAW MRI → CLEAN TRAIN DATASET
# GPU/CPU optimized + clean + readable
# ============================================================

import os
import pandas as pd
import numpy as np
import nibabel as nib
import dicom2nifti
from nilearn import image, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import multiprocessing

# ================= GPU/CPU BOOST =================
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

# ================= PATH =================
RAW_MRI_DIR = "MRI"
CSV_FILE = "mri_metadata.csv"
OUTPUT_DIR = "prossed_dataset"
TARGET_SHAPE = (128,128,128)

# ================= LOAD TEMPLATE =================
print("Loading MNI template...")
MNI_TEMPLATE = datasets.load_mni152_template()
GM_MASK = datasets.load_mni152_gm_mask()

# ============================================================
# SEGMENT + REGISTER
# ============================================================
def segment_register(nifti_path):

    img = nib.load(nifti_path)

    # register to MNI
    registered = image.resample_to_img(img, MNI_TEMPLATE)

    # grey matter mask
    gm = image.math_img("img * mask", img=registered, mask=GM_MASK)

    return gm

# ============================================================
# CROP OR PAD
# ============================================================
def crop_or_pad(data):

    x,y,z = data.shape
    tx,ty,tz = TARGET_SHAPE

    out = np.zeros(TARGET_SHAPE)

    cx,cy,cz = x//2,y//2,z//2
    ox,oy,oz = tx//2,ty//2,tz//2

    xs=max(cx-ox,0); ys=max(cy-oy,0); zs=max(cz-oz,0)
    xe=min(xs+tx,x); ye=min(ys+ty,y); ze=min(zs+tz,z)

    crop=data[xs:xe,ys:ye,zs:ze]
    out[:crop.shape[0],:crop.shape[1],:crop.shape[2]] = crop

    return out

# ============================================================
# NORMALIZE
# ============================================================
def normalize(data):
    mn,mx=data.min(),data.max()
    return (data-mn)/(mx-mn+1e-8)

# ============================================================
# PROCESS ONE MRI
# ============================================================
def process_one(item):

    sid=item["id"]
    label=item["label"]
    dicom_path=item["path"]
    split=item["split"]

    try:
        save_dir=os.path.join(OUTPUT_DIR,split,label)
        os.makedirs(save_dir,exist_ok=True)

        temp_nii=f"temp_{sid}.nii.gz"

        dicom2nifti.dicom_series_to_nifti(
            dicom_path,
            temp_nii,
            reorient_nifti=True
        )

        gm=segment_register(temp_nii)
        data=gm.get_fdata()

        data=crop_or_pad(data)
        data=normalize(data)

        final_name=f"{sid}_clean.nii.gz"
        save_path=os.path.join(save_dir,final_name)

        final=nib.Nifti1Image(data,MNI_TEMPLATE.affine)
        nib.save(final,save_path)

        if os.path.exists(temp_nii):
            os.remove(temp_nii)

    except Exception as e:
        print("Error:",sid,e)

# ============================================================
# MAIN PIPELINE
# ============================================================
def run():

    print("\nStarting full preprocessing...")

    # reset output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # ================= LOAD CSV =================
    df=pd.read_csv(CSV_FILE)
    df.columns=[c.lower().strip() for c in df.columns]

    label_map={}
    for _,r in df.iterrows():
        label_map[str(r["subject"]).strip()] = str(r["group"]).strip()

    print("Subjects in CSV:",len(label_map))

    # ================= FIND MRI =================
    data=[]

    for root,dirs,files in os.walk(RAW_MRI_DIR):
        if files:
            for sid in label_map:
                if sid in root:
                    data.append({
                        "id":sid,
                        "path":root,
                        "label":label_map[sid]
                    })
                    break

    print("MRI found:",len(data))

    # ================= SPLIT =================
    labels=[d["label"] for d in data]

    train,temp = train_test_split(
        data,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    val,test = train_test_split(
        temp,
        test_size=0.5,
        stratify=[d["label"] for d in temp],
        random_state=42
    )

    for d in train: d["split"]="training"
    for d in val: d["split"]="validation"
    for d in test: d["split"]="testing"

    all_data=train+val+test

    print(f"""
DATA SUMMARY
Train: {len(train)}
Val:   {len(val)}
Test:  {len(test)}
""")

    # ================= MULTIPROCESS (FAST) =================
    print("\nProcessing MRI (parallel)...")

    pool = multiprocessing.Pool(processes=8)
    list(tqdm(pool.imap(process_one, all_data), total=len(all_data)))
    pool.close()
    pool.join()

    print("\n====================================")
    print("MRI PREPROCESSING COMPLETED")
    print("Output folder:",OUTPUT_DIR)
    print("====================================")

# ============================================================
if __name__=="__main__":
    run()
