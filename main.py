from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import os
from pathlib import Path
import shutil
from typing import List, Tuple

import cv2
from tqdm import tqdm
from PIL import Image
import pyheif
from termcolor import colored


cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")


class FaceImageClassifier:
    def __init__(self, path: str) -> None:
        absolute_path = os.path.abspath(path)
        self.check_existence(absolute_path)

        target_directory_path = os.path.join(absolute_path, "**/*")
        paths = glob(target_directory_path, recursive=True)
        self.file_paths = filter(lambda path: os.path.isfile(path), paths)

        self.valid_extensions = [".heic", ".png", ".jpg", ".jpeg"]

        self.detected_image_paths = []

    def move(self, move_directory_path: str) -> FaceImageClassifier:
        self.check_existence(move_directory_path)
        for detected_image_path in self.detected_image_paths:
            shutil.move(detected_image_path, move_directory_path)
        return self

    def copy(self, copy_directory_path: str) -> FaceImageClassifier:
        self.check_existence(copy_directory_path)
        for detected_image_path in self.detected_image_paths:
            shutil.copy(detected_image_path, copy_directory_path)
        return self

    def _get_image_paths(self) -> List[Tuple(str, str)]:
        image_paths = []
        for file_path in self.file_paths:
            extension = Path(file_path).suffix.lower()
            if extension not in self.valid_extensions:
                continue
            image_paths.append((file_path, extension))
        return image_paths

    def search(self, **kwargs) -> FaceImageClassifier:
        image_paths = self._get_image_paths()
        with tqdm(total=len(image_paths)) as progress:
            with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                futures = []
                for image_path, extension in image_paths:
                    future = executor.submit(
                        FaceImageClassifier.process_image,
                        image_path,
                        extension,
                        **kwargs,
                    )
                    future.add_done_callback(lambda _: progress.update())
                    futures.append(future)
                result = [future.result() for future in futures if future.result()]
            self.detected_image_paths = result
        return self

    @staticmethod
    def detect_image(
        image_path: str, is_rectangle_enabled: bool = False, **kwargs
    ) -> bool:
        image = cv2.imread(image_path)
        face_list = cascade.detectMultiScale(image, minSize=(150, 150))
        if len(face_list) == 0:
            return False

        if is_rectangle_enabled:
            for x, y, w, h in face_list:
                color = (0, 0, 255)
                thickness = 3
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=thickness)
            cv2.imwrite(image_path, image)

        return True

    @staticmethod
    def process_image(image_path: str, extension: str, **kwargs) -> str:
        if extension == ".heic":
            try:
                image_path = FaceImageClassifier.heif_to_png(image_path, **kwargs)
            except ValueError as e:
                colored_image_path = colored(image_path, "blue")
                print(f"{colored_image_path}: {e}")
                return ""
        face_included = FaceImageClassifier.detect_image(image_path, **kwargs)
        return image_path if face_included else ""

    @staticmethod
    def check_existence(path: str) -> None:
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"{path} does not exist")

    @staticmethod
    def heif_to_png(image_path: str, is_deleted: bool = False, **kwargs) -> str:
        heif_file = pyheif.read(image_path)
        png_path = image_path.replace(".HEIC", ".png")

        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

        image.save(png_path, "PNG")
        if is_deleted:
            os.remove(image_path)

        return png_path


if __name__ == "__main__":
    face_image_classifier = FaceImageClassifier(path="./target")
    face_image_classifier.search(is_deleted=True).move("./out")
