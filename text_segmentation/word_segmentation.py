import numpy as np

from text_segmentation.utils import trim_360, trim_section


def segment_words(section, vertical_projection):
    whitespace_lengths = []
    whitespace = 0

    # A) Get whitespace lengths in the more dense subsection of a section, hence ranging from 5 to len(v_p)-4, to avoid
    # the bias of long ascenders and descenders
    for idx in range(5, len(vertical_projection) - 4):
        if vertical_projection[idx] == 0:
            whitespace = whitespace + 1
        elif vertical_projection[idx] != 0:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)
            whitespace = 0  # reset whitespace counter.
        if idx == len(vertical_projection) - 1:
            whitespace_lengths.append(whitespace)
    # print("whitespaces:", whitespace_lengths)
    avg_white_space_length = np.mean(whitespace_lengths)
    # print("average whitespace lenght:", avg_white_space_length)

    # B) Find words with whitespaces which are actually long spaces (word breaks) using the avg_white_space_length
    whitespace_length = 0
    divider_indexes = [0]
    for index, vp in enumerate(vertical_projection[4:len(vertical_projection) - 5]):
        if vp == 0:  # white
            whitespace_length += 1
        elif vp != 0:  # black
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indexes.append(index - int(whitespace_length / 2))
            whitespace_length = 0  # reset it
    divider_indexes.append(len(vertical_projection) - 1)
    divider_indexes = np.array(divider_indexes)
    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))
    new_dividers = [window for window in dividers if np.sum(
        np.sum(section[:, window[0]:window[1]], axis=0)) > 200]

    return new_dividers


def word_segmentation(section_images):
    words_in_sections = []  # |-------- pad obtained sections
    sections = []
    for idx in range(len(section_images)):
        section = trim_section(section_images[idx])
        if section.shape[0] == 0 or section.shape[1] == 0:
            continue
        sections.append(section)

        vertical_projection = np.sum(section, axis=0)
        dividers = segment_words(section, vertical_projection)
        words = []
        for window in dividers:
            word = section[:, window[0]:window[1]]
            trimmed_word = trim_360(word)
            # plotSimpleImages(sliding_words[-1])
            words.append(trimmed_word)
        words_in_sections.append(words)
        images = [section, vertical_projection]
        images.extend(words)
        # plotGrid(images_boolean_to_binary)
        # plotGrid(images)
    print("Word segmentation complete.")
    return sections, words_in_sections