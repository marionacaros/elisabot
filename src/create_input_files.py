## Copyright 2019
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from model_VQG.utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='VQG',
                       json_path='//work/mcaros/TFM/data/dict/dataset.json',
                       image_folder='//projects/language/vqa2019/VQG/',
                       captions_per_image=5,
                       min_word_freq=2,
                       output_folder='./data/caption_data/',
                       max_len=6)
