import numpy as np


def calc_outlier(data, method="std"):
    '''method used to find outlier of list, either std or iqr used'''
    if method == "iqr":
        # method1: interquartile
        q3, q1 = np.percentile(data, [75, 25])
        iqr = q3 - q1
        outlier = q1 - 1.5 * iqr
    else:
        # method2: standard deviation
        outlier = np.mean(data) - 1.5*np.std(data)
    return outlier

def trim_section(section, section_threshold=10):
    """ Function for removing padding from left and right side of a character section """
    vertical_projection = np.sum(section, axis=0)
    b1 = 0
    b2 = 0
    beginning = 0
    end = 0
    temp1 = 0
    temp2 = 0

    for idx in range(len(vertical_projection)):
        if beginning == 0:
            if vertical_projection[idx] == 0:  # white
                if b1 <= section_threshold:
                    temp1 = 0
                    b1 = 0
            elif vertical_projection[idx] != 0:  # black
                if b1 == 0:  # start of black
                    temp1 = idx - 1 if idx - 1 > 0 else idx
                if b1 > section_threshold:
                    beginning = temp1
                b1 += 1

        if end == 0:
            idx2 = len(vertical_projection) - (idx + 1)
            if vertical_projection[idx2] == 0:  # white
                if b2 <= section_threshold:
                    temp2 = 0
                    b2 = 0
            elif vertical_projection[idx2] != 0:  # black

                if b2 == 0:  # start of black
                    temp2 = idx2 + 1 if idx + \
                                        1 < len(vertical_projection) else idx2
                if b2 > 10:
                    end = temp2
                b2 += 1

        if end != 0 and beginning != 0:
            break

    new_section = section[:, beginning:end]
    return new_section.astype(np.uint8)


def trim_360(image, section_thresh=5):
    """ Returns an image with no padding on either side """
    trim1 = trim_section(np.rot90(image).astype(int), section_threshold=section_thresh)
    trim2 = trim_section(np.rot90(trim1, axes=(1, 0)).astype(int), section_threshold=section_thresh - 5)
    return trim2