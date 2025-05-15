import pandas as pd
import os

# Load endpoints TSV
endpoints_path = '/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/CMR_phenotypes_final.tsv'
endpoints_df = pd.read_csv(endpoints_path, sep='\t')

# Get patient IDs from directory
patient_dirs_path = '/projects/prjs1252/data_jesse/cmr_cropped'
available_patients = [d for d in os.listdir(patient_dirs_path) 
                     if os.path.isdir(os.path.join(patient_dirs_path, d))]

# Convert to strings for consistent comparison
patient_id_col = 'f.eid'  # Change this to match your actual patient ID column
endpoints_df[patient_id_col] = endpoints_df[patient_id_col].astype(str)
available_patients = [str(p) for p in available_patients]

# Find missing patients
tsv_patients = set(endpoints_df[patient_id_col])
directory_patients = set(available_patients)
missing_patients = directory_patients - tsv_patients

# Print missing patient stats
print(f"Total patients in directories: {len(directory_patients)}")
print(f"Patients missing from TSV file: {len(missing_patients)} ({len(missing_patients)/len(directory_patients):.1%})")


# Filter endpoints to only include available patients
filtered_df = endpoints_df[endpoints_df[patient_id_col].isin(available_patients)]

# Select only columns of interest
endpoints_of_interest = [
    patient_id_col,   # Keep patient ID column
    'LVEF',          # Left Ventricular Ejection Fraction 
]
filtered_df = filtered_df[endpoints_of_interest]

# Save to CSV
output_dir = os.path.join("/projects/prjs1252/data_jesse/metadata")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "filtered_endpoints.csv")
filtered_df.to_csv(output_path, index=False)

print(f"Saved {len(filtered_df)} patients with {len(endpoints_of_interest)} endpoints to {output_path}")