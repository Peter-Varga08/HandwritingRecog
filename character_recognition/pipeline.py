import copy

import cv2
import numpy as np
import torch

from character_recognition.model import CharacterRecognizer
from text_segmentation.character_segmentation import slide_over_word, select_slides, destructure_characters, clean_image
from text_segmentation.pipeline import SegmentationPipeline


class RecognitionPipeline:
    def __init__(self):
        self.name2idx = {
            "Alef": 0,
            "Ayin": 1,
            "Bet": 2,
            "Dalet": 3,
            "Gimel": 4,
            "He": 5,
            "Het": 6,
            "Kaf": 7,
            "Kaf-final": 8,
            "Lamed": 9,
            "Mem": 10,
            "Mem-medial": 11,
            "Nun-final": 12,
            "Nun-medial": 13,
            "Pe": 14,
            "Pe-final": 15,
            "Qof": 16,
            "Resh": 17,
            "Samekh": 18,
            "Shin": 19,
            "Taw": 20,
            "Tet": 21,
            "Tsadi-final": 22,
            "Tsadi-medial": 23,
            "Waw": 24,
            "Yod": 25,
            "Zayin": 26,
        }
        self.segmentation_pipeline = SegmentationPipeline()
        self.model = CharacterRecognizer()
        self.model.load_model(checkpoint='40_char_rec.ckpt')

    def __call__(self, img):
        characters_word_line, char_widths, mean_char_width = self.segmentation_pipeline(img)
        window_size = int(mean_char_width) if int(mean_char_width) > 90 else 90
        shift = 1
        all_segmented_characters = []
        all_segmented_labels = []
        all_char_propabilities = []
        characters_skipped = 0
        for line in characters_word_line:
            line_imgs = []
            line_labels = []
            for word in line:
                word_imgs = []
                word_labels = []
                for char_idx, character_segment in enumerate(word):
                    if character_segment.shape[1] > mean_char_width + np.std(
                            char_widths
                    ):  # multiple characters suspected
                        try:
                            predicted_char_num = round(
                                character_segment.shape[1] / mean_char_width
                            )
                            sliding_characters = slide_over_word(
                                character_segment, window_size, shift
                            )
                            (
                                recognised_characters,
                                predicted_labels,
                                probabilities,
                            ) = select_slides(
                                sliding_characters,
                                predicted_char_num,
                                self.model,
                                window_size,
                                self.name2idx,
                            )
                            multiple_characters = copy.deepcopy(recognised_characters)
                            multiple_characters.append(character_segment)
                            predictions_string = ""
                            for label in predicted_labels:
                                predictions_string = (
                                    f"{predictions_string}, {list(self.name2idx.keys())[label]}"
                                )
                            word_imgs.extend(recognised_characters)
                            word_labels.extend(predicted_labels)
                            all_char_propabilities.extend(probabilities)
                        except:
                            continue

                    else:  # single character
                        if character_segment.size != 0:
                            character_segment = clean_image(character_segment)
                            predicted_label, probability = self.model.predict(character_segment)
                            predicted_letter = list(self.name2idx.keys())[predicted_label]
                            word_imgs.append(character_segment)
                            word_labels.append(predicted_label)
                            all_char_propabilities.append(probability)
                        else:
                            characters_skipped += 1
                line_imgs.append(word_imgs[::-1])
                line_labels.append(word_labels[::-1])
            all_segmented_characters.append(line_imgs[::-1])
            all_segmented_labels.append(line_labels[::-1])

        # for character in destructure_characters(all_segmented_characters):
        #     plot_simple_images([character], title='hello')

        labels_for_file = copy.deepcopy(all_segmented_labels)

        # work with these for style classification, they are in one array
        all_segmented_labels = np.asarray(destructure_characters(all_segmented_labels))
        all_segmented_characters = destructure_characters(all_segmented_characters)
        all_char_propabilities = np.asarray(all_char_propabilities)

        return all_segmented_characters, all_char_propabilities, labels_for_file

