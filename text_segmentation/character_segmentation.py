import copy
import itertools
from typing import List

import cv2
import numpy as np

from character_recognition.model import CharacterRecognizer
from text_segmentation.utils import calc_outlier, trim_360


class CharacterSegmentation:
    def __init__(self):
        pass

    def __call__(self, words_in_lines):
        return self.run_pipeline(words_in_lines)

    def run_pipeline(self, words_in_lines):
        '''
        entire pipleline for character segmentation, takes words in all lines obtained from the scroll and
        returns list of charcters in words in lines
        '''
        # Get all characters from all words
        segmented_word_box_images, segmented_word_box_areas, all_box_boundaries = self.run_character_segment(
            words_in_lines)
        filtered_word_box_images_all_lines, character_widths = self.filter_characters(segmented_word_box_areas,
                                                                                      segmented_word_box_images,
                                                                                      all_box_boundaries,
                                                                                      words_in_lines)
        single_character_widths = [width for width in character_widths if width <= 150 and width >= 20]
        mean_single_character_width = np.mean(single_character_widths)

        all_suspected = 0
        changed = 0
        characters_eroded = []
        for line in filtered_word_box_images_all_lines:
            eroded_lines = []
            for word in line:
                eroded_words = []
                for char_idx, character_segment in enumerate(word):
                    if character_segment.shape[1] > mean_single_character_width + np.std(single_character_widths):
                        all_suspected += 1
                        # Run connected components to get number of labels, so merged clusters are identified beforehand
                        character_segment = character_segment.astype(np.uint8)
                        num_labels, clusters = cv2.connectedComponents(character_segment, connectivity=4)
                        clusters = self.get_component_clusters(num_labels, clusters)
                        eroded_img_boundaries, eroded_img = self.erode_clusters(character_segment, (2, 2), iter_num=6)
                        eroded_box_img_list, eroded_box_areas, eroded_img = self.get_box_images(eroded_img_boundaries,
                                                                                                eroded_img)
                        filtered_eroded_box_img_list = self.dilate_and_filter_eroded_characters(
                            segmented_word_box_areas,
                            eroded_box_areas,
                            eroded_box_img_list,
                            eroded_img_boundaries,
                            eroded_img)
                        eroded_words.extend(filtered_eroded_box_img_list)
                    else:
                        eroded_words.append(character_segment)
                eroded_lines.append(eroded_words)
            characters_eroded.append(eroded_lines)
        # print(f'all:{all_suspected}, changed:{changed}')
        return characters_eroded, single_character_widths, mean_single_character_width

    def slide_over_word(self, word, window_size, shift):
        '''function that takes an image of a word and then slides a window of a given size over and obtaines snapshots
        the window is shifted by the value shift until it reaches the end of the word
        returns these images'''
        images = []
        height, width = word.shape
        for snap in range(0, width - window_size, shift):
            images.append(word[:, snap:snap + window_size])
            # plotSimpleImages([word[:, snap:snap + window_size]])
        images.append(word[:, word.shape[1] - window_size: word.shape[1]])
        return images

    def get_sliding_words(self, words_in_lines, window_size, shift):
        '''takes a list containing the images of all word in the lines of an scroll
        calls sliding words on these words and returns an array in the same structure'''
        sliding_words_in_line = []
        for line in words_in_lines:
            sliding_words = []
            for word in line:
                sliding_words.append(self.slide_over_word(word, window_size, shift))
            sliding_words_in_line.append(sliding_words)
        return sliding_words_in_line

    def clean_image(self, image, thresh_side=500, thresh_mid=20, trim_thresh=4, skip_left_pruning=False,
                    skip_right_pruning=False):
        '''function to remove artifacts, removes small artifacts contained in an image under a certain pixel count, larger ones that touch
        the border of the image in question. finally the borders of the image are removed'''
        # image = get_bina'''ry(image)
        image = image.astype(np.uint8)
        new = self.remove_character_artifacts(image, skip_left_pruning=skip_left_pruning,
                                              skip_right_pruning=skip_right_pruning, internal_min_cluster=thresh_mid)
        new = trim_360(new, section_thresh=trim_thresh)
        if new.size == 0:
            new = image
        return new

    def get_component_clusters(self, num_labels, labels):
        '''function that takes a list of pixel locations (tuple of x, y locations) that indicate a connected component
        returns a numpy array representing an image of the connected component'''
        clusters = [[] for _ in range(num_labels)]
        for row_idx, row in enumerate(labels):
            for col_idx, col in enumerate(row):
                clusters[col].append([row_idx, col_idx])
        del clusters[0]
        return clusters

    def get_bounding_box_boundaries(self, image, clusters) -> List[List[list]]:
        '''places a bounding box around a connected component so it can be turned into a numpy array'''
        box_boundaries = []
        for idx, cluster in enumerate(clusters):
            # initialize starting values
            y_max, y_min, x_max, x_min = [image.shape[0], 0, 0, image.shape[1]]

            for coordinate in cluster:
                if coordinate[0] < y_max:
                    y_max = coordinate[0]
                elif coordinate[0] > y_min:
                    y_min = coordinate[0]
                if coordinate[1] > x_max:
                    x_max = coordinate[1]
                elif coordinate[1] < x_min:
                    x_min = coordinate[1]
            if x_max != image.shape[1]:
                x_max += 1
            if x_min != 0:
                x_min -= 1
            box_boundaries.append([[y_max, y_min], [x_min, x_max]])
        return box_boundaries

    def dilate_clusters(self, word, kernel=(5, 3)):
        '''function that dilates a word with the given kernel, returns the bounding boxes of the connected components after
        dilation'''
        kernel = np.ones(kernel, np.uint8)
        word = cv2.dilate(word, kernel, iterations=1)
        num_labels, clusters = cv2.connectedComponents(word, connectivity=4)
        clusters = self.get_component_clusters(num_labels, clusters)
        box_boundaries = self.get_bounding_box_boundaries(word, clusters)
        box_boundaries = sorted(box_boundaries, key=lambda x: x[1][1])
        # plotConnectedComponentBoundingBoxes(word, box_boundaries)
        return box_boundaries, word

    def get_box_images(self, box_boundaries, word):
        """Returns all the bounded images and areas thereof within a word -- that can contain any number of characters """
        new_word = copy.deepcopy(word)
        new_word = np.insert(new_word, 0, 0, axis=1)
        box_images = []
        box_areas = []
        for box in box_boundaries:
            y_min = box[0][0]
            y_max = box[0][1]
            x_min = box[1][0] - 1 if box[1][0] - 1 != -1 else 0
            x_max = box[1][1] + 1
            box_img = new_word[y_min:y_max, x_min:x_max]
            box_areas.append(abs(y_max - y_min) * abs(x_max - x_min))
            box_images.append(box_img)
        return box_images, box_areas, new_word

    def erode_clusters(self, word, kernel, iter_num=1):
        '''Erodes a words determined by the kernel and the iter num. Orderes the resulting bounding boxes of the
        resulting connected componentents and returns them'''
        kernel = np.ones(kernel, np.uint8)
        word = cv2.erode(word, kernel, iterations=iter_num)
        num_labels, clusters = cv2.connectedComponents(word, connectivity=4)
        clusters = self.get_component_clusters(num_labels, clusters)
        box_boundaries = self.get_bounding_box_boundaries(word, clusters)
        box_boundaries = sorted(box_boundaries, key=lambda x: x[1][1])
        # plotConnectedComponentBoundingBoxes(word, box_boundaries)
        return box_boundaries, word

    def character_segment(self, word, title=None):
        '''takes a word and takes the connected components in that word
        if the word has many artifacts resulting in a high number of connecting components
        dilate until their amount reaches a threshold'''
        cluster_threshold = 7
        word = word.astype(np.uint8)
        num_labels, clusters = cv2.connectedComponents(word, connectivity=4)
        clusters = self.get_component_clusters(num_labels, clusters)
        box_boundaries = self.get_bounding_box_boundaries(word, clusters)
        # sort characters from left to right
        box_boundaries = sorted(box_boundaries, key=lambda x: x[1][1])
        num_boxes = len(box_boundaries)
        while num_boxes > cluster_threshold:
            box_boundaries, word = self.dilate_clusters(word)
            num_boxes = len(box_boundaries)
            # print(num_boxes)
        # erosion, character segment, dialate clusters

        box_images, box_areas, new_word = self.get_box_images(box_boundaries, word)
        # plot_connected_component_bounding_boxes(word, box_boundaries, title = title)
        # print("Character segmentation complete.")
        # plot_simple_images(box_images)
        return box_images, box_areas, new_word, box_boundaries

    def run_character_segment(self, words_in_lines):
        '''runs character segment pipelin on a list of words in each line of a scroll
        returns the images of each character in each word in each line of a scroll'''
        segmented_word_box_images = []
        segmented_word_box_areas = []
        all_box_boundaries = []
        count = 0
        for line_idx, line in enumerate(words_in_lines):
            line_word_images = []
            line_word_areas = []
            box_boundaries_lines = []
            for word_idx, word in enumerate(line):
                if word.size != 0:
                    box_images, areas, word, box_boundaries = self.character_segment(word,
                                                                                     title="[OLD]")  # have them here
                    words_in_lines[line_idx][word_idx] = word
                    line_word_images.append(box_images)
                    line_word_areas.append(areas)
                    box_boundaries_lines.append(box_boundaries)
                    count += len(box_boundaries_lines)
            segmented_word_box_images.append(line_word_images)
            segmented_word_box_areas.append(line_word_areas)
            all_box_boundaries.append(box_boundaries_lines)
        print('run character segment:', count)
        return segmented_word_box_images, segmented_word_box_areas, all_box_boundaries

    def is_boundary_included(self, all_boundries, cluster):
        '''checks if the max and min of a bounding box of a cluster is contained within a bounding box of another character
        this would means that the cluster is obsolete and can be added to the cluster that it is contained in'''
        x_min_input = cluster[1][0]
        x_max_input = cluster[1][1]
        for idx, boundries in enumerate(all_boundries):
            x_min_current = boundries[1][0]
            x_max_current = boundries[1][1]
            if x_min_input > x_min_current and x_max_input < x_max_current:
                return True
        return False

    def is_image_border_active(self, character):
        '''checks whether the left or right most column of a character image contain 1s'''
        left_border = character[:, 0]
        right_border = character[:, -1]
        if len(np.nonzero(left_border)[0]) > 0 or len(np.nonzero(right_border)[0]) > 0:
            return True
        return False

    def get_character_area_outlier(self, segmented_word_box_areas):
        '''function to determine whether a character is an outlier or not'''
        # Empirically observed values
        min_area = 500
        max_area = 8000  # anything above 8000 is undoubtedly more than 1 character in any test image

        area_thr = lambda img, x: [x for x in img if x > min_area and x < max_area]
        boxes_thr = [[area_thr(image, None) for image in line if area_thr(image, None) != []] for line in
                     segmented_word_box_areas]
        flat_boxes_thr = list(itertools.chain(*list(itertools.chain(*boxes_thr))))
        # Filter artifacts that are still small, but large enough to be mistaken for words, where filtering is based on
        # the average word size of the current document
        outlier_thr = calc_outlier(flat_boxes_thr)
        if outlier_thr < min_area:
            outlier_thr = min_area
        return outlier_thr

    def filter_characters(self, segmented_word_box_areas, segmented_word_box_images, all_box_boundaries,
                          words_in_lines):
        """
        Returns those images that are supposedly not artifacts of whole set of words in lines.
        """

        outlier_thr = self.get_character_area_outlier(segmented_word_box_areas)
        char_num = 0
        for line in segmented_word_box_images:
            for word in line:
                for char in word:
                    char_num += 1
        # print("Number of characters before filtering:", char_num)

        filtered_word_box_images = []
        character_widths = []
        for i, line in enumerate(segmented_word_box_images):
            line_list = []
            for j, word in enumerate(line):
                if word != []:
                    word_list = []
                    for k, character in enumerate(word):
                        if character.size != 0:
                            skip_left_pruning = False
                            skip_right_pruning = False
                            if segmented_word_box_areas[i][j][k] >= outlier_thr:
                                new_cluster = words_in_lines[i][j]
                                boundries = all_box_boundaries[i][j][k]
                                if not self.is_boundary_included(all_box_boundaries[i][j], boundries):
                                    x_min = boundries[1][0] - 1 if boundries[1][0] - 1 != -1 else 0
                                    x_max = boundries[1][1] + 1
                                    taller_cluster = new_cluster[:, x_min:x_max]
                                    if x_min == 0:
                                        skip_left_pruning = True
                                    if x_max == new_cluster.shape[1]:
                                        skip_right_pruning = True
                                    if taller_cluster != []:
                                        # plot_simple_images([character, taller_cluster, clean_image(taller_cluster)])
                                        if self.is_image_border_active(taller_cluster):
                                            word_list.append(
                                                self.clean_image(character, skip_left_pruning=skip_left_pruning,
                                                                 skip_right_pruning=skip_right_pruning))
                                        else:
                                            word_list.append(
                                                self.clean_image(taller_cluster, skip_left_pruning=skip_left_pruning,
                                                                 skip_right_pruning=skip_right_pruning))
                                        character_widths.append(word_list[-1].shape[1])
                    if word_list != []:
                        line_list.append(word_list)
            if line_list != []:
                filtered_word_box_images.append(line_list)
        # print('filter characters',len(filtered_word_box_images))
        char_num = 0
        for line in filtered_word_box_images:
            for word in line:
                for char in word:
                    char_num += 1
        # print("*"*40)
        # print("Number of characters after filtering:", char_num)
        # print("*"*40)
        return filtered_word_box_images, character_widths

    def remove_character_artifacts(self, image, skip_left_pruning=False, skip_right_pruning=False,
                                   internal_min_cluster=30):
        '''function that will locate and remove artifacts of an image'''
        img_copy = copy.deepcopy(image)
        num_labels, labels = cv2.connectedComponents(img_copy, connectivity=4)
        clusters = self.get_component_clusters(num_labels, labels)
        sizes = []
        # print('----')
        for cluster in clusters:
            # print(np.sum(cluster))
            sizes.append(np.sum(cluster))
        # print('----')

        min_cluster = np.mean(sizes)
        if len(clusters) > 1:
            left_border = img_copy[:, 0]
            right_border = img_copy[:, -1]
            for cluster in clusters:
                if np.sum(cluster) < min_cluster:
                    # clean artifacts at the borders
                    for y, x in cluster:
                        if (x == 0 and left_border[y] and not skip_left_pruning
                            or x == img_copy.shape[1] - 1 and right_border[y] and not skip_right_pruning) \
                                and img_copy[y, x]:
                            for y, x in cluster:
                                img_copy[y, x] = 0
                            break
                    # clean artifacts inside
                    if np.sum(cluster) < internal_min_cluster:
                        for y, x in cluster:
                            for y, x in cluster:
                                img_copy[y, x] = 0
                            break
            # plotSimpleImages([img_copy, image])
        return img_copy

    def destructure_characters(self, characters_in_line):
        '''takes a list ordered lsit of lines of words of characters to a single list'''
        characters = []
        for line in characters_in_line:
            for word in line:
                for character in word:
                    characters.append(character)
        return characters

    def select_slides(self, sliding_characters, predicted_char_num, model: CharacterRecognizer, window_size, name2idx):
        '''
        given a set of slides obtained by sliding over a word, determine which slides are most likely to be an actual character
        or not using the trained model
        '''
        shift = 1
        chosen_characters = 2
        first = trim_360(sliding_characters[0])
        last = trim_360(sliding_characters[-1])
        first_label, prob_first = model.predict(first)
        last_label, prob_last = model.predict(last)

        recognised_characters = [first]
        probabilities = [prob_first]
        labels = [first_label]
        prev_letter_start = 0
        start_idx = 0
        while chosen_characters < predicted_char_num:
            best_prob = 0
            chosen_slide = []
            chosen_label = 0
            for idx, slide in enumerate(sliding_characters[start_idx:]):
                if slide.size[1] != 0:
                    start = shift * idx
                    end = start + window_size
                    begin_limit = int(prev_letter_start + window_size * 0.75)
                    end_limit = int(prev_letter_start + window_size * 0.75 + window_size + window_size * 0.6)
                    # print(begin_limit, end_limit)
                    if start >= begin_limit and end <= end_limit:
                        predicted_label, probability = model.predict(slide)
                        predicted_letter = list(name2idx.keys())[predicted_label]
                        # print(f'Predicted label:{predicted_letter} probabilty:{probability}')
                        # print(f"window: [{shift * idx}-{window_size + shift * idx}]")
                        if probability > best_prob:
                            best_prob = probability
                            chosen_slide = slide
                            chosen_label = predicted_label
                            temp_idx = idx
            chosen_characters += 1
            if len(chosen_slide) != 0:
                recognised_characters.append(self.clean_image(chosen_slide))
                probabilities.append(best_prob)
                start_idx = temp_idx
            labels.append(chosen_label)
            prev_letter_start = 0

        recognised_characters.append(last)
        probabilities.append(prob_last)
        labels.append(last_label)
        return recognised_characters, labels, probabilities

    def dilate_and_filter_eroded_characters(self, segmented_word_box_areas, eroded_box_areas, eroded_box_img_list,
                                            eroded_box_boundaries,
                                            eroded_img):
        '''
        erode characters suspected to contain multiple characters to attempt to split them up
        filter them and return the newly seperated characters
        '''
        outlier_thr = self.get_character_area_outlier(segmented_word_box_areas)
        # plot_simple_images([eroded_img], title= 'before')
        filtered_images = []
        for i, image in enumerate(eroded_box_img_list):
            if image.size != 0:
                skip_left_pruning = False
                skip_right_pruning = False
                if eroded_box_areas[i] >= outlier_thr:
                    boundary = eroded_box_boundaries[i]
                    if not self.is_boundary_included(eroded_box_boundaries, boundary):
                        x_min = boundary[1][0] - 1 if boundary[1][0] - 1 != -1 else 0
                        x_max = boundary[1][1] + 1
                        taller_cluster = eroded_img[:, x_min:x_max]
                        if x_min == 0:
                            skip_left_pruning = True
                        if x_max == eroded_img.shape[1]:
                            skip_right_pruning = True
                        if taller_cluster != []:
                            kernel = np.ones((2, 2), np.uint8)
                            if self.is_image_border_active(taller_cluster):
                                image = np.pad(image, pad_width=20, mode="constant", constant_values=0)
                                dilated_img = cv2.dilate(image, kernel, iterations=6)
                                filtered_images.append(
                                    self.clean_image(dilated_img, skip_left_pruning=skip_left_pruning,
                                                     skip_right_pruning=skip_right_pruning))
                            else:
                                taller_cluster = np.pad(taller_cluster, pad_width=20, mode="constant",
                                                        constant_values=0)
                                dilated_img = cv2.dilate(taller_cluster, kernel, iterations=6)
                                filtered_images.append(
                                    self.clean_image(dilated_img, skip_left_pruning=skip_left_pruning,
                                                     skip_right_pruning=skip_right_pruning))
        return filtered_images
