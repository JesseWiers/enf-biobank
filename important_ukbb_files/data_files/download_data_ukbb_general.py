# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    The script downloads the cardiac MR images for a UK Biobank Application and
    converts the DICOM into nifti images.
    """
import os
import glob
import pandas as pd
import shutil
from biobank_utils import *
import dateutil.parser
import random
import logging
import time
import argparse


if __name__ == '__main__':
    
    # Directory to save the created dataset in 

    logging.getLogger().setLevel(logging.INFO)

    out_path = '/projects/prjs1252/dataset'

    parser = argparse.ArgumentParser(description="Process UK Biobank MRIs and convert DICOM to NIfTI.")
    
    parser.add_argument(
        '--eid_file',
        type=str,
        required=True,
        help='csv file with f.eids of interest'
    )

    parser.add_argument(
        '--out_path',
        type=str,
        required=True,
        help='directory to save the nifti files in'
    )
    
    args = parser.parse_args()
    
    eid_file = args.eid_file

    data_list = pd.read_csv(eid_file, header=None)[0].astype(str).tolist()[1:]
    logging.info(len(data_list))

    # Initialize a list to collect metadata from all patients
    all_patients_metadata = []
    
    # Creating the dataset
    start_idx = 0
    end_idx = len(data_list)
    
    for i in range(start_idx, end_idx):

        # Start the timer
        start_time = time.time()

        eid = str(data_list[i])
        
        logging.info(f"\033[33m Processing patient with eid: {eid} \033[0m")
        logging.info(f"i={i}")

        # Check if DICOM files exist
        dicom_zip = os.path.join(out_path, '/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20209_short_axis_heart/imaging_visit_array_0/{0}_20209_2_0.zip'.format(eid))
        if not os.path.exists(dicom_zip):
            logging.warning(f"Missing DICOM zip file for EID: {eid}")
            continue  # Skip this patient
        
        # Patient directory
        data_dir = os.path.join(out_path, eid)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # # Patient DICOM directory
        dicom_dir = os.path.join(data_dir, 'dicom')
        if not os.path.exists(dicom_dir):
            os.mkdir(dicom_dir)
     
        # 209 -> short axis
        shutil.copy(os.path.join(out_path, '/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20209_short_axis_heart/imaging_visit_array_0/{0}_20209_2_0.zip'.format(eid)), dicom_dir)
        
        # Unpack the data (list the zip files for the different modalities)
        files = glob.glob('{1}/{0}_*.zip'.format(eid, dicom_dir))  

        # Files just contains a zip file for SAX now        
        for f in files:
            
            # logging.info(f"Unzipping file: {f}")
                        
            # Actually unzip the file ('>' part is to suppress the output)
            os.system('unzip -o {0} -d {1} > /dev/null 2>&1'.format(f, dicom_dir))
                        
            # Convert the cvs file to csv, since cvs doesn't make any sense    
            if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
                os.system('cp {0} {1}'.format(os.path.join(dicom_dir, 'manifest.cvs'),
                                              os.path.join(dicom_dir, 'manifest.csv')))
                
            # process (just removes comma's) the manifest file and write it to manifest2.csv

            try:
                process_manifest(os.path.join(dicom_dir, 'manifest.csv'),
                                os.path.join(dicom_dir, 'manifest2.csv'))
            except Exception as e:
                logging.error(f"Error processing manifest for EID {eid}: {str(e)}")
                continue  # Skip this patient if manifest processing fails
            
            # read the manifest2.csv file into a dataframe
            df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), on_bad_lines='skip')
            
            # Patient ID and acquisition date
            pid = df2.at[0, 'patientid']
            date = dateutil.parser.parse(df2.at[0, 'date'][:11]).date().isoformat()
            
            # Group the files into subdirectories for each imaging series
            
            for series_name, series_df in df2.groupby('series discription'):
                
                # Creates a directory for each series, lovely
                # Series refers to different z's or InlineVF (Ventricular Function)
                series_dir = os.path.join(dicom_dir, series_name)
                if not os.path.exists(series_dir):
                    os.mkdir(series_dir)
                    
                # Get the filenames for each series and move them to the series directories
                series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
                os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))
                
            
        # # Convert dicom files and annotations into nifti images
        
        # Create a Biobank_Dataset object based on all the series for a particular patient
        # It practically justs creates dictionaries for different series subclasses (SAX, LAX)
        try:
            dset = Biobank_Dataset(dicom_dir)
            dset.read_dicom_images()
            dset.convert_dicom_to_nifti(data_dir)
        except Exception as e:
            logging.error(f"Error converting DICOM to NIfTI for EID {eid}: {str(e)}")
            continue  # Skip this patient if DICOM to NIfTI conversion fails
       
        # Get metadata for this patient
        patient_metadata = dset.get_metadata()
        
        # Add patient ID to each metadata entry
        for entry in patient_metadata:
            entry['Patient ID'] = eid
        
        # Add to the collection of all patients
        all_patients_metadata.extend(patient_metadata)

        # Remove intermediate files
        os.system('rm -rf {0}'.format(dicom_dir))
        os.system('rm -f {0}_*.zip'.format(eid))

        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"Patient took {elapsed:.2f} seconds")
        
    try:
        metadata_path = os.path.join(out_path, 'all_patients_metadata.xlsx')
        
        # Create DataFrame from new batch metadata
        df_new = pd.DataFrame(all_patients_metadata)

        if not df_new.empty:
            cols = ['Patient ID'] + [col for col in df_new.columns if col != 'Patient ID']
            df_new = df_new[cols]

            # If Excel exists, load and merge
            if os.path.exists(metadata_path):
                logging.info("Loading existing metadata file to append...")
                df_old = pd.read_excel(metadata_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                
                # Remove duplicates (if Patient ID is unique)
                df_combined = df_combined.drop_duplicates(subset=['Patient ID'])
            else:
                df_combined = df_new

            # Save back to Excel
            df_combined.to_excel(metadata_path, index=False)
            logging.info(f"\033[32m Metadata for all patients exported to {metadata_path} \033[0m")
        else:
            logging.info("No new metadata to save.")

    except ImportError:
        logging.info("Error: pandas is required for Excel export. Install with 'pip install pandas openpyxl'")
        
        