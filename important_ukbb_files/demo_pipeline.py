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
    This script demonstrates a pipeline for cardiac MR image analysis.
"""
import os
import urllib.request
import shutil

if __name__ == '__main__':
    # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --save_seg False --seq_name sa --data_dir /projects/prjs1252/data_jesse_final_v3/nifti_dataset --model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))