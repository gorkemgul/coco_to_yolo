import json
import os
import time
import shutil
import yaml
from tqdm import tqdm
from addict import Dict
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCOConverter():
    """
    Converts labels from COCO format to YOLO (seg or detection)

    Args:
         coco_annot_path (str): COCO annotation path to be converted
         converted_label_path (str): Label path for converted labels to be saved
         image_path (str): Image directory path where images are located
         conversion_mode (str): Label output mode to decide whether as segmentation or detection
         yolo_train_path (str): Path to save train labels
         yolo_test_path (str): Path to save test labels
         yolo_val_path (str): Path to save validation labels
         train_ratio (float): Ratio of images to be used for training
         valid_ratio (float): Ratio of images to be used for validation
         test_ratio (float): Ratio of images to be used for testing
         split_data (bool): Whether to perform train-test-validation split
         dataset_name (str): The name of the dataset to organize the output directories
    """

    def __init__(self, coco_annot_path: str, converted_label_path: str, image_path: str, conversion_mode: str,
                 yolo_train_path: str, yolo_test_path: str, yolo_val_path: str,
                 train_ratio: float = 0.80, valid_ratio: float = 0.10, test_ratio: float = 0.10,
                 split_data: bool = False, dataset_name: str = "dataset"):
        super().__init__()
        self.coco_annot_path = coco_annot_path
        self.converted_label_path = converted_label_path
        self.image_path = image_path
        self.conversion_mode = conversion_mode
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.split_data = split_data
        self.yolo_train_path = yolo_train_path
        self.yolo_test_path = yolo_test_path
        self.yolo_val_path = yolo_val_path
        self.dataset_name = dataset_name

        if not (0 <= train_ratio <= 1 and 0 <= valid_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1")
        if abs(train_ratio + valid_ratio + test_ratio - 1) > 1e-6:
            raise ValueError("The sum of train_ratio, valid_ratio, and test_ratio must be 1")

        # Ensure the labels directory exists
        self.labels_path = os.path.join(self.converted_label_path, "labels")
        if not os.path.exists(self.labels_path):
            logger.info(f"Creating 'labels' directory at {self.labels_path}")
            os.makedirs(self.labels_path)

    def from_coco_to_yolo(self):
        try:
            with open(self.coco_annot_path, 'r') as json_file:
                data_dict = json.load(json_file)
        except FileNotFoundError:
            logger.error(f"COCO annotation file not found: {self.coco_annot_path}")
            return
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON file: {self.coco_annot_path}")
            return

        if not os.path.exists(self.converted_label_path):
            os.makedirs(self.converted_label_path)

        data = Dict(data_dict)
        image_info = data["images"]
        annotations = data["annotations"]
        self.class_ids = {}
        self.class_names = []

        start_time = time.time()
        for i_info in tqdm(image_info, desc="Processing images"):
            lines = []

            for anno in annotations:
                if i_info['id'] == anno['image_id']:
                    class_id = anno['category_id'] - 1
                    if class_id not in self.class_ids:
                        self.class_ids[class_id] = len(self.class_names)
                        self.class_names.append(data["categories"][anno['category_id'] - 1]['name'])

                    line = str(self.class_ids[class_id])
                    line += " "

                    try:
                        if self.conversion_mode == "detection":
                            x, y, w, h = anno['bbox']
                            x_centre = (x + (x + w)) / 2
                            y_centre = (y + (y + h)) / 2
                            line += f"{x_centre / i_info['width']} {y_centre / i_info['height']} {w / i_info['width']} {h / i_info['height']}"
                        elif self.conversion_mode == "segmentation":
                            for idx, seg in enumerate(anno["segmentation"][0]):
                                if idx % 2 == 0:
                                    line += f"{seg / i_info['width']} "
                                else:
                                    line += f"{seg / i_info['height']} "
                        else:
                            logger.error("Invalid conversion mode! Choose 'detection' or 'segmentation'.")
                            continue
                    except KeyError as e:
                        logger.error(f"Missing key in annotation: {e}")
                        continue

                    lines.append(line)

            label_name = os.path.join(self.labels_path, f"{i_info['file_name'].split('.')[0]}.txt")
            with open(label_name, 'w') as f:
                for line in lines:
                    f.write(line + '\n')

        end_time = time.time()
        logger.info(f"Process completed in {end_time - start_time} seconds")

        self.create_config_yaml()

    def train_test_valid_split(self):
        start_time = time.time()
        train_dest_path = os.path.join(self.converted_label_path, "train")
        test_dest_path = os.path.join(self.converted_label_path, "test")
        valid_dest_path = os.path.join(self.converted_label_path, "valid")

        for path in [train_dest_path, test_dest_path, valid_dest_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                os.makedirs(os.path.join(path, "images"))
                os.makedirs(os.path.join(path, "labels"))

        images = [i for i in os.listdir(self.image_path) if i.lower().endswith((".jpeg", ".jpg", ".png"))]
        total_images = len(images)

        # Calculate the number of images for each set
        train_count = round(self.train_ratio * total_images)
        valid_count = round((self.valid_ratio * total_images))
        test_count = total_images - (train_count + valid_count)

        # Ensure the total number of images adds up to the original count
        remaining_images = total_images - (train_count + valid_count + test_count)
        if remaining_images > 0:
            test_count += remaining_images

        # Split images
        train_images = images[:train_count]
        valid_images = images[train_count:train_count + valid_count]
        test_images = images[train_count + valid_count:train_count + valid_count + test_count]

        logger.info(
            f"Dataset split: {len(train_images)} train, {len(valid_images)} validation, {len(test_images)} test")

        # Move images and labels to respective directories
        for split, image_list in zip(["train", "test", "valid"], [train_images, test_images, valid_images]):
            for image in tqdm(image_list, desc=f"Processing {split} images"):
                src_image = os.path.join(self.image_path, image)
                dst_image = os.path.join(self.converted_label_path, split, "images", image)
                src_label = os.path.join(self.labels_path, f"{image.rsplit('.', 1)[0]}.txt")
                dst_label = os.path.join(self.converted_label_path, split, "labels", f"{image.rsplit('.', 1)[0]}.txt")

                shutil.copy(src_image, dst_image)
                if os.path.exists(src_label):
                    shutil.copy(src_label, dst_label)

        end_time = time.time()
        logger.info(f"Image separation completed in {end_time - start_time} seconds")

        # Move directories and config.yaml to the dataset_name directory
        self.move_to_dataset_directory()

    def create_config_yaml(self):
        config_path = os.path.join(self.converted_label_path, "config.yaml")

        if not os.path.exists(config_path):
            logger.info(f"Creating configuration file at {config_path}")

            # Determine the paths for train, test, and validation
            yolo_train_path = os.path.join(self.converted_label_path, "train")
            yolo_test_path = os.path.join(self.converted_label_path, "test")
            yolo_val_path = os.path.join(self.converted_label_path, "valid")

            # Collect class names and ensure they are indexed by class ID
            class_names = {i: name for i, name in enumerate(self.class_names)}

            config_data = {
                'train': yolo_train_path,
                'test': yolo_test_path,
                'val': yolo_val_path,
                'nc': len(class_names),
                'names': class_names
            }

            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False)
                logger.info(f"Configuration file created successfully at {config_path}")
        else:
            logger.info(f"Configuration file already exists at {config_path}")

    def move_to_dataset_directory(self):
        dataset_dir = os.path.join(self.converted_label_path, self.dataset_name)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Move the directories and config.yaml to the dataset directory
        for split in ["train", "test", "valid"]:
            split_dir = os.path.join(self.converted_label_path, split)
            if os.path.exists(split_dir):
                shutil.move(split_dir, os.path.join(dataset_dir, split))

        config_file = os.path.join(self.converted_label_path, "config.yaml")
        if os.path.exists(config_file):
            shutil.move(config_file, dataset_dir)

        logger.info(f"Moved dataset directories and config.yaml to {dataset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to YOLO format and optionally split dataset.")
    parser.add_argument('--coco-annot-path', required=True, type=str, help="Path to COCO annotations JSON file")
    parser.add_argument('--image-path', required=True, type=str, help="Path to directory containing images")
    parser.add_argument('--converted-label-path', required=True, type=str, help="Path to save converted labels")
    parser.add_argument('--conversion-mode', required=True, choices=['detection', 'segmentation'],
                        help="Conversion mode: 'detection' or 'segmentation'")
    parser.add_argument('--train-ratio', type=float, default=0.80, help="Ratio of images to use for training")
    parser.add_argument('--valid-ratio', type=float, default=0.15, help="Ratio of images to use for validation")
    parser.add_argument('--test-ratio', type=float, default=0.05, help="Ratio of images to use for testing")
    parser.add_argument('--yolo-train-path', required=True, type=str, help="Path to save train labels")
    parser.add_argument('--yolo-test-path', required=True, type=str, help="Path to save test labels")
    parser.add_argument('--yolo-val-path', required=True, type=str, help="Path to save validation labels")
    parser.add_argument('--split-data', type=bool, default=False,
                        help="Whether to split dataset into train, test, and validation")
    parser.add_argument('--dataset-name', required=True, type=str, help="Name of the dataset directory")

    args = parser.parse_args()

    converter = COCOConverter(
        coco_annot_path=args.coco_annot_path,
        image_path=args.image_path,
        converted_label_path=args.converted_label_path,
        conversion_mode=args.conversion_mode,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        yolo_train_path=args.yolo_train_path,
        yolo_test_path=args.yolo_test_path,
        yolo_val_path=args.yolo_val_path,
        split_data=args.split_data,
        dataset_name=args.dataset_name
    )

    converter.from_coco_to_yolo()

    if args.split_data:
        converter.train_test_valid_split()
