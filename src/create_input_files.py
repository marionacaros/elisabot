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
