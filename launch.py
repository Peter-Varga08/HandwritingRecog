import argparse
import warnings
from pathlib import Path
from pprint import pprint as pp
from typing import List, Union

from PIL.Image import Image
from tqdm import tqdm

from character_recognition.pipeline import RecognitionPipeline
from style_classification.pipeline import StyleClassificationPipeline
from font2image import labeltotext, styletotext

warnings.filterwarnings("ignore")


def run_pipelines(img_path: Union[Path, Image], results_path: str = "results"):

    characters, probabilities, labels = RecognitionPipeline()(img_path)
    print("*" * 40)
    print("TOTAL NUMBER OF CHARACTERS:", len(characters))
    print("*" * 40)

    print("Getting style classification for all chararcters:")
    dominant_style = StyleClassificationPipeline()(characters, probabilities)

    # save results
    Path(results_path).mkdir(parents=True, exist_ok=True)
    img_name = img_path.stem
    labeltotext(labels, img_name)
    styletotext(dominant_style, img_name)
    print(f"Image {img_name} has been processed successfully.\n")


def main(test_images: List[Path]) -> None:
    """
    Main function to run the pipeline on all images in a folder.

    :param test_images: Name of each jpg image in the folder.
    """
    for img_path in tqdm(test_images):
        print(f"Running Handwriting Recognition on image {img_path}")
        run_pipelines(img_path=img_path)
        # try:
        #     run_pipeline(img_path=img_path)
        #
        #     break
        # except Exception as error:
        #     print("Fatal Error:", error)
        #     print(f"Skipping image [{img_path}]")


if __name__ == "__main__":
    # data/cropped_labeled_images/
    # parsing path for image folder
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to folder with image files")
    args = parser.parse_args()

    # get input path of folder and separate img names w/ glob
    test_folder = args.path
    if test_folder is None:
        test_folder = "data/image-data/binary"
    test_folder = Path(test_folder)
    test_images = list(test_folder.glob("*.jpg"))
    print("TEST IMAGES:")
    pp(test_images)
    main(test_images=test_images)
