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
# ============================================================================
import os
import time
import math
import numpy as np
import nibabel as nib
import tensorflow as tf
import argparse
import sys
import pandas as pd
sys.path.append('/home/jwiers/deeprisk')
from ukbb_cardiac.common.image_utils import rescale_intensity
from tqdm import tqdm  # Import tqdm for progress tracking
import logging


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Define your flags (arguments)
parser.add_argument('--seq_name', type=str, default='sa', choices=['sa', 'la_2ch', 'la_4ch'], help='Sequence name.')
parser.add_argument('--data_dir', type=str, default='/projects/prjs1252/data_jesse_final_v3/nifti_dataset', help='Path to the data set directory.')
parser.add_argument('--model_path', type=str, default='trained_model/FCN_sa', help='Path to the saved trained model.')
parser.add_argument('--process_seq', type=bool, default=True, help='Process a time sequence of images.')
parser.add_argument('--save_seg', type=bool, default=True, help='Save segmentation.')
parser.add_argument('--seg4', type=bool, default=False, help='Segment all the 4 chambers in long-axis 4 chamber view.')

# Parse the arguments
FLAGS = parser.parse_args()

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))
        saver.restore(sess, '{0}'.format(FLAGS.model_path))

        logging.info('Start deployment on the data set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(FLAGS.data_dir))
        processed_list = []
        table_time = []

        csv_path = '/projects/prjs1252/data_jesse_final_v3/metadata/filtered_endpoints.csv'

        if os.path.exists(csv_path):
            # Load existing file
            df = pd.read_csv(csv_path)
            logging.info(f"✅ Loaded existing CSV with {len(df)} rows.")
        else:
            # Create empty DataFrame
            df = pd.DataFrame(columns=['f.eid', 'ED', 'ES'])
            logging.info("✅ Created new empty DataFrame.")

        # logging.info("FLAGS.process_seq:, ", FLAGS.process_seq)
        # print("data_list = ", data_list)
    
        for (idx, data) in enumerate(tqdm(data_list, desc="Processing Data", unit="dataset")):
            # print("data = ", data)
            logging.info(f"idx: {idx}")
            data_dir = os.path.join(FLAGS.data_dir, data)


            seg_file = os.path.join(data_dir, f"seg_{FLAGS.seq_name}.nii.gz")

            if os.path.exists(seg_file):
                logging.info(f"Skipping {data_dir} — segmentation already exists.")
                continue

            image_name = '{0}/{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
            patient_id = os.path.basename(os.path.dirname(image_name))

            # if patient_id not in df['f.eid'].astype(str).values:
            #     print(f"Skipping patient_id {patient_id}, not in metadata.")
            #     continue
        
            # print("data_dir = ", data_dir)

            if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                seg_name = '{0}/seg4_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
            else:
                seg_name = '{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)

            if FLAGS.process_seq:
                
                # Process the temporal sequence
                # print("image_name: ", image_name)
                
                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file '
                          'name {1}. Skip.'.format(data_dir, os.path.basename(image_name)))
                    continue

                # Read the image
                # print('Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                image = nim.get_fdata()
                X, Y, Z, T = image.shape
                orig_image = image

                # print('Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))

                # Prediction (segmentation)
                pred = np.zeros(image.shape)

                # Pad the image size to be a factor of 16 so that the
                # downsample and upsample procedures in the network will
                # result in the same image size at each resolution level.
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                for t in range(T):
                    # Transpose the shape to NXYC
                    image_fr = image[:, :, :, t]
                    image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                    image_fr = np.expand_dims(image_fr, axis=-1)

                    # Evaluate the network
                    prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                                feed_dict={'image:0': image_fr, 'training:0': False})

                    # Transpose and crop segmentation to recover the original size
                    pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                    pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                    pred[:, :, :, t] = pred_fr

                seg_time = time.time() - start_seg_time
                logging.info('  Segmentation time = {:3f}s'.format(seg_time))
                table_time += [seg_time]
                processed_list += [data]

                # ED frame defaults to be the first time frame.
                # Determine ES frame according to the minimum LV volume.
                k = {}
                k['ED'] = 0
                if FLAGS.seq_name == 'sa' or (FLAGS.seq_name == 'la_4ch' and FLAGS.seg4):
                    k['ES'] = int(np.argmin(np.sum(pred == 1, axis=(0, 1, 2))))
                    k['ED'] = int(np.argmax(np.sum(pred == 1, axis=(0, 1, 2))))

                # Convert patient_id to string (just in case)
                patient_id_str = str(patient_id)

                # Check if this f.eid already exists
                if patient_id_str in df['f.eid'].astype(str).values:
                    # Update existing row
                    df.loc[df['f.eid'].astype(str) == patient_id_str, 'ED'] = k['ED']
                    df.loc[df['f.eid'].astype(str) == patient_id_str, 'ES'] = k['ES']
                else:
                    # Append new row
                    df = pd.concat([
                        df,
                        pd.DataFrame({
                            'f.eid': [patient_id_str],
                            'ED': [k['ED']],
                            'ES': [k['ES']]
                        })
                    ], ignore_index=True)


                if (idx%500==0): 
                    logging.info("saving csv file")
                    df.to_csv(csv_path, index=False)

                # Save the segmentation
                if FLAGS.save_seg:
                    print('  Saving segmentation ...')
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                        seg_name = '{0}/seg4_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
                    else:
                        seg_name = '{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
                        
                    print(f"saving segmentation to {seg_name}")
                    nib.save(nim2, seg_name)

                    for fr in ['ED', 'ES']:
                        nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                                 '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr))
                        if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                            seg_name = '{0}/seg4_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        else:
                            seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine), seg_name)
            else:
                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with '
                          'file name {1} or {2}. Skip.'.format(data_dir,
                                                               os.path.basename(image_ED_name),
                                                               os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)

                    # Read the image
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    image = nim.get_data()
                    X, Y = image.shape[:2]
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=2)

                    print('Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity rescaling
                    image = rescale_intensity(image, (1, 99))

                    # Pad the image size to be a factor of 16 so that
                    # the downsample and upsample procedures in the network
                    # will result in the same image size at each resolution
                    # level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    logging.info('Segmentation time = {:3f}s'.format(seg_time))
                    table_time += [seg_time]
                    processed_list += [data]

                    # Save the segmentation
                    if FLAGS.save_seg:
                        print('  Saving segmentation ...')
                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                            seg_name = '{0}/seg4_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        else:
                            seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        nib.save(nim2, seg_name)

        # SAVE UPDATED CSV FILE (DATAFRAME)
        df.to_csv(csv_path, index=False)
        
        if FLAGS.process_seq:
            logging.info('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
        else:
            logging.info('Average segmentation time = {:.3f}s per frame'.format(np.mean(table_time)))
        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
            process_time, len(processed_list), process_time / len(processed_list)))
