from pathlib import Path
from typing import Union

from PIL.Image import Image

from text_segmentation.character_segmentation import CharacterSegmentation
from text_segmentation.line_segmentation import line_segmentation
from text_segmentation.word_segmentation import word_segmentation


class SegmentationPipeline:
    def __init__(self) -> None:
        pass

    def __call__(self, img_path: ...):
        return self.run_pipeline(img_path=img_path)

    def run_pipeline(self, img_path: Union[Path, Image]):
        if isinstance(img_path, Path):
            aStar_path = f"paths/{img_path.with_suffix('')}"
        print("Running line segmentation...")
        section_images = line_segmentation(str(img_path), aStar_path)
        print("Running word segmentation...")
        lines, words_in_lines = word_segmentation(section_images)
        print("Running character segmentation...")
        character_segmentation = CharacterSegmentation()
        characters_word_line, single_character_widths, mean_character_width = character_segmentation(words_in_lines)
        return characters_word_line, single_character_widths, mean_character_width