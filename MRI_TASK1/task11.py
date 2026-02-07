import os
import pandas as pd
import nibabel as nib
import numpy as np
import dicom2nifti
from nilearn import image, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

RAW_MRI_DIR = "MRI"
METADATA_CSV = "mri_metadata.csv"
OUTPUT_DIR = "processed_dataset"
TARGET_SHAPE = (128, 128, 128)

MNI_TEMPLATE = datasets.load_mni152_template()
GM_MASK = datasets.load_mni152_gm_mask()

def segment_and_register(nifti_img):
    #  grey matter
    # ---------Skull stripping and Spatial normalization-------------
    resampled = image.resample_to_img(nifti_img, MNI_TEMPLATE)
    gm_img = image.math_img("img * mask", img=resampled, mask=GM_MASK)
    return gm_img
# resizing
def center_crop_or_pad(img_data):
   
    c_x, c_y, c_z = np.array(img_data.shape) // 2
    t_x, t_y, t_z = np.array(TARGET_SHAPE) // 2
    
 
    cropped = img_data[c_x-t_x:c_x+t_x, c_y-t_y:c_y+t_y, c_z-t_z:c_z+t_z]
    return cropped

def min_max_scaling(data):
    # ========= intensity normalization=================
    d_min, d_max = data.min(), data.max()
    denom = (d_max - d_min) + 1e-8
    return (data - d_min) / denom

def run_pipeline():
    # Refresh the output directory for a clean run
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare metadata and remove subject
    df = pd.read_csv(METADATA_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates(subset=['subject'], keep='first')
    label_map = {str(k).strip(): str(v).strip() for k, v in zip(df['subject'], df['group'])}

    # Map the unique subjects to their physical folder locations
    data_list = []
    for root, dirs, files in os.walk(RAW_MRI_DIR):
        if files:
            for sid in label_map.keys():
                if sid in root:
                    data_list.append({'id': sid, 'path': root, 'label': label_map[sid]})
                    break
    
    data_list = list({v['id']: v for v in data_list}.values())
    

    train_data, temp_data = train_test_split(data_list, test_size=0.3, stratify=[d['label'] for d in data_list])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=[d['label'] for d in temp_data])

    dataset_splits = {
        'training': train_data,
        'validation': val_data,
        'testing': test_data
    }

    print(f"Total subjects: {len(data_list)}. (Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)})")

    for split_name, items in dataset_splits.items():
        print(f"Processing {split_name} set...")
        for item in tqdm(items):
            try:
                # Create directory structure: processed_dataset/[split]/[label]
                dest = os.path.join(OUTPUT_DIR, split_name, item['label'])
                os.makedirs(dest, exist_ok=True)

                #  Convert the raw DICOM folder to a temporary NIfTI file
                temp_nii = f"temp_{item['id']}.nii.gz"
                dicom2nifti.dicom_series_to_nifti(item['path'], temp_nii, reorient_nifti=True)

            
                gm_segmented_img = segment_and_register(temp_nii)
                
              
                raw_data = gm_segmented_img.get_fdata()
                cropped_data = center_crop_or_pad(raw_data)

               
                final_data = min_max_scaling(cropped_data)

                final_name = f"{item['id']}_clean.nii.gz"
                output_file = os.path.join(dest, final_name)
                final_nii = nib.Nifti1Image(final_data, MNI_TEMPLATE.affine)
                nib.save(final_nii, output_file)
                
                # Remove temp file to clear disk space
                if os.path.exists(temp_nii): 
                    os.remove(temp_nii)

            except Exception as e:
                print(f"Error processing subject {item['id']}: {e}")

    print(f"Pipeline finished! Data is ready in '{OUTPUT_DIR}' for Training, Validation, and Testing.")

if __name__ == "__main__": 
    run_pipeline()