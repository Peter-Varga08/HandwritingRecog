import pickle

import numpy as np

from character_recognition.utils import resize_pad
from style_classification.SVM_Style import get_hog


class StyleClassificationPipeline:
    def __init__(self):
        pass

    def __call__(self, characters, probabilities, prob_threshold=0.8):
        '''
        Runs the SVM for each character that is nicely segmented (high probability from the recognizer)
        Returns the style of that image based on simple majority voting
        '''
        # load the model from disk
        filename = 'SVM_for_Style.sav'
        clf = pickle.load(open(filename, 'rb'))

        print('#Segmented characters from pipeline')
        print(len(characters))

        index = np.where(probabilities <= prob_threshold)
        characters = np.delete(characters, index)

        print('#Number of characters after removal by probability thresholding')
        print(len(characters))

        characters_on_page = []
        for char in characters:
            resized_char = resize_pad(char, 40, 40, 0)
            characters_on_page.append(np.asarray(resized_char, dtype=float))

        characters_on_page = np.asarray(get_hog(characters_on_page))

        predicted_styles = clf.predict(characters_on_page)

        archaic_counter = len(predicted_styles[predicted_styles == 0])
        hasmonean_counter = len(predicted_styles[predicted_styles == 1])
        herodian_counter = len(predicted_styles[predicted_styles == 2])

        print("Individual Characters Style prediction counter")
        print('Archaic: ', archaic_counter, 'Hasmonean: ', hasmonean_counter, 'Herodian: ', herodian_counter)

        pred_style = max(archaic_counter, hasmonean_counter, herodian_counter)

        if pred_style == archaic_counter: return 'Archaic'
        if pred_style == hasmonean_counter: return 'Hasmonean'
        if pred_style == herodian_counter: return 'Herodian'
