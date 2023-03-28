from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import os
from pathlib import Path
import shutil
from typing import List, Tuple

import cv2
import dlib
from PIL import Image
import pyheif
from tqdm import tqdm


class FaceImageClassifier:
    def __init__(self, path: str) -> None:
        self.detected_image_paths = []
        self.valid_extensions = [".heic", ".png", ".jpg", ".jpeg"]

        absolute_path = os.path.abspath(path)
        self.check_existence(absolute_path)
        target_directory_path = os.path.join(absolute_path, "**/*")
        paths = glob(target_directory_path, recursive=True)
        self.file_paths = filter(lambda path: os.path.isfile(path), paths)

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

    def classifier(self, **kwargs) -> FaceImageClassifier:
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
    def is_face_included(
        image_path: str, is_rectangle_enabled: bool = False, **kwargs
    ) -> bool:
        detector = dlib.get_frontal_face_detector()

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            return False

        if is_rectangle_enabled:
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(image_path, image)

        return True

    @staticmethod
    def process_image(image_path: str, extension: str, **kwargs) -> str:
        if extension == ".heic":
            try:
                image_path = FaceImageClassifier.heif_to_png(image_path, **kwargs)
            except ValueError:
                return ""
        face_included = FaceImageClassifier.is_face_included(image_path, **kwargs)
        return image_path if face_included else ""

    @staticmethod
    def check_existence(path: str) -> None:
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"{path} does not exist")

    @staticmethod
    def heif_to_png(image_path: str, is_image_deleted: bool = False, **kwargs) -> str:
        png_path = image_path.replace(".HEIC", ".png")
        if os.path.exists(png_path):
            return png_path

        heif_file = pyheif.read(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

        image.save(png_path, "PNG")
        if is_image_deleted:
            os.remove(image_path)

        return png_path


if __name__ == "__main__":
    face_image_classifier = FaceImageClassifier(path="./target")
    face_image_classifier.classifier(
        is_rectangle_enabled=True, is_image_deleted=True
    ).move("./out")
