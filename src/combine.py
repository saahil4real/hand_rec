import os
import sys
import argparse
import json
from typing import Tuple, List

import cv2
import pylev
from PIL import Image
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from split_line import *
from format_change import *
from spell_check import *
from test import *


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = 'charList.txt'
    fn_summary = 'summary.json'
    fn_corpus = 'corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        # run model on validation data.
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = pylev.levenshtein(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path) -> None:
	"""Recognizes text in image provided by file path."""

	# to store all predictions
	findings = []

	all_files = []
	# format change --> obj detection --> call infer

	# convert to numpy array and store in a folder.
	files = convert_to_array(fn_img)

	# i=0
	# object detection and converting to grayscale
	for file in files:
		arr = obj_detection(file)
		for x in arr:
			x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
			if(x.shape[0] >= 10 and x.shape[1] >= 10):
				all_files.append(x)
			
	# now predicting text on all these single line images
	for file in all_files:
		# image_path = base_path + file

		# img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		# assert img is not None

		img = file

		preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
		img = preprocessor.process_img(img)

		batch = Batch([img], None, 1)
		recognized, probability = model.infer_batch(batch, True)
		#run spell check on predicted statement
		recognized = correct_sentence(recognized[0])
		findings.append(recognized)
		# findings.append(recognized[0])

		

	for finding in findings:
		print(finding)
	return findings


# def main():
#     """Main function."""
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
#     # parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
#     parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
#     parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
#     parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
#     parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
#     parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
#     parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
#     args = parser.parse_args()

#     # set chosen CTC decoder
#     # decoder_mapping = {'bestpath': DecoderType.BestPath,
#     #                    'beamsearch': DecoderType.BeamSearch,
#     #                    'wordbeamsearch': DecoderType.WordBeamSearch}
#     decoder_type = DecoderType.BestPath

#     # train or validate on IAM dataset
#     if args.mode in ['train', 'validate']:
#         # load training data, create TF model
#         loader = DataLoaderIAM(args.data_dir, args.batch_size)
#         char_list = loader.char_list

#         # when in line mode, take care to have a whitespace in the char list
#         if args.line_mode and ' ' not in char_list:
#             char_list = [' '] + char_list

#         # save characters of model for inference mode
#         open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

#         # save words contained in dataset into file
#         open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

#         # execute training or validation
#         if args.mode == 'train':
#             model = Model(char_list, decoder_type)
#             train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)
#         elif args.mode == 'validate':
#             model = Model(char_list, decoder_type, must_restore=True)
#             validate(model, loader, args.line_mode)

#     # infer text on test image
#     elif args.mode == 'infer':
#         model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)
#         infer(model, args.img_file)


# if __name__ == '__main__':
#     main()

