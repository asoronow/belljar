import numpy as np
import tifffile as tiff
import os
import cv2
import shutil
import yolov5

def unsharp_mask(img):
    mask = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )

    return cv2.filter2D(img, -1, mask, borderType=cv2.BORDER_CONSTANT)

def slice_to_squares(input_dir, size=640, output_dir=""):
    for file in os.listdir(input_dir):
        # if its a tif load it
        if file.endswith(".tif") or file.endswith(".tiff"):
            # chop into squares
            img = tiff.imread(os.path.join(input_dir, file))
            # find the z dimension, the smallest dimension
            z = np.argmin(img.shape)
            # max project the image
            img = np.max(img, axis=z)
            # apply an unsharp mask
            img = unsharp_mask(img)
            # chop into squares
            tile_num = 0
            for i in range(0, img.shape[1], size):
                for j in range(0, img.shape[0], size):
                    if i + size > img.shape[1] or j + size > img.shape[0]:
                        continue
                    cv2.imwrite(
                        os.path.join(
                            output_dir,
                            file.replace(".ome.tiff", f"_{tile_num}.png"),
                        ),
                        img[j : j + size, i : i + size],
                    )
                    tile_num += 1

def make_dataset(tile_dir, output_dir, model_path, confidence=0.5):
    '''
    Makes a 80/20 split train test dataset in the Yolo format in the output dir.

    File Structure:
        output_dir/
            images/
                train/
                val/
            labels/
                train/
                val/
            dataset.yaml

    Args:
        tile_dir (str): Directory of tiles
        output_dir (str): Directory to output dataset
        model_path (str): Path to model
        confidence (float): Confidence threshold
    '''

    # make the directories
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # get the files
    files = os.listdir(tile_dir)
    files = [f for f in files if f.endswith(".png")]
    # shuffle the files
    np.random.shuffle(files)
    # split the files
    train_files = files[: int(len(files) * 0.8)]
    val_files = files[int(len(files) * 0.8) :]
    # get the model
    detection_model = yolov5.load(model_path, device="cuda:0")
    detection_model.conf = confidence
    # loop through the files
    for t_c, img in enumerate(train_files):
        print(f"Processing {t_c}/{len(train_files)} of train files...", end="\r")
        # img path
        img_path = os.path.join(tile_dir, img)

        img_data = cv2.imread(img_path)

        # convert to 8 bit
        img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        results = detection_model(img_data)

        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        classes = predictions[:, 5]

        labels = []

        for box, score, c in zip(boxes, scores, classes):
            if score < 0.9 or img_data is None:
                continue
            x1, y1, x2, y2 = box
            img_width, img_height = img_data.shape[1], img_data.shape[0]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1

            # normalized xywh
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            annotation = f"{int(c)} {x_center} {y_center} {width} {height}"
            labels.append(annotation)
        
        # DEBUG: Show a random annotation on the image to confirm coordinates
        # if len(labels) > 0:
        #     label = labels[0]
        #     c, x_center, y_center, width, height = label.split(" ")
        #     x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
        #     x1, y1, x2, y2 = int((x_center - width / 2) * img_width), int((y_center - height / 2) * img_height), int((x_center + width / 2) * img_width), int((y_center + height / 2) * img_height)

        #     cv2.rectangle(img_data, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     cv2.imshow("img", img_data)
        #     cv2.waitKey(0)

        if len(labels) == 0:
            continue

        # copy the image to the train folder
        cv2.imwrite(os.path.join(output_dir, "images", "train", img), img_data)
        # write the labels to the train folder
        with open(os.path.join(output_dir, "labels", "train", img.replace(".png", ".txt")), "w") as f:
            f.write("\n".join(labels))
        
    for t_v, img in enumerate(val_files):
        print(f"Processing {t_v}/{len(val_files)} of val files...", end="\r")
        # img path
        img_path = os.path.join(tile_dir, img)

        img_data = cv2.imread(img_path)

        # convert to 8 bit
        img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        results = detection_model(img_data)

        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        classes = predictions[:, 5]

        labels = []

        for box, score, c in zip(boxes, scores, classes):
            if score < 0.9 or img_data is None:
                continue
            x1, y1, x2, y2 = box
            img_width, img_height = img_data.shape[1], img_data.shape[0]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1

            # normalized xywh
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            annotation = f"{int(c)} {x_center} {y_center} {width} {height}"
            labels.append(annotation)
        

        if len(labels) == 0:
            continue

        # copy the image to the train folder
        cv2.imwrite(os.path.join(output_dir, "images", "val", img), img_data)
        # write the labels to the train folder
        with open(os.path.join(output_dir, "labels", "val", img.replace(".png", ".txt")), "w") as f:
            f.write("\n".join(labels))

    # write the dataset.yaml
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(
            f"""
                train: {os.path.join(output_dir, 'images', 'train')}
                val: {os.path.join(output_dir, 'images', 'val')}
                nc: 1
                names: ['neuron']
            """
        )


tile_dir = r"\\128.114.78.227\euiseokdataUCSC_1\Matt Jacobs\Images and Data\M Brains\M322\02 - counting\tiles"
output_dir = r"\\128.114.78.227\euiseokdataUCSC_1\Matt Jacobs\Images and Data\M Brains\M322\02 - counting\for_training"
more_tiles_dir=r"\\128.114.78.227\euiseokdataUCSC_1\Matt Jacobs\Images and Data\M Brains\M286\Counting\02 - rabies"
# slice_to_squares(more_tiles_dir, output_dir=tile_dir)
# make_dataset(tile_dir, output_dir, r"C:\Users\imageprocessing\.belljar\models\ancientwizard.pt")

drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1
curr_img = None
boxes = []
img_copy = None

def inspect_dataset(path_to_images):
    '''Read in and inspect a dataset to confirm or modify bounding boxes, or delete junk images.'''

    # Get the images and their full paths and infer labels from the file names
    train_images = [os.path.join(path_to_images, "images", "train", f) for f in os.listdir(os.path.join(path_to_images, "images", "train")) if f.endswith(".png")]
    val_images = [os.path.join(path_to_images, "images", "val", f) for f in os.listdir(os.path.join(path_to_images, "images", "val")) if f.endswith(".png")]
    train_labels = [os.path.join(path_to_images, "labels", "train", f) for f in os.listdir(os.path.join(path_to_images, "labels", "train")) if f.endswith(".txt")]
    val_labels = [os.path.join(path_to_images, "labels", "val", f) for f in os.listdir(os.path.join(path_to_images, "labels", "val")) if f.endswith(".txt")]

    marked_for_deletion = []

    for img_path, img_label in zip([*train_images, *val_images], [*train_labels, *val_labels]):
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        # load the labels
        with open(img_label, "r") as f:
            labels = f.readlines()
            # convert normalized xywh to x1y1x2y2
            print(f"Processing {img_path} @ {img_label}")
            for label in labels:
                label = label.split(" ")
                c, x_center, y_center, width, height = label

                # multiply by the image dimensions
                x_center, y_center, width, height = float(x_center) * img_width, float(y_center) * img_height, float(width) * img_width, float(height) * img_height
                x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow("Image", img)
            
        should_continue = False
        while not should_continue:
            action = cv2.waitKey(0)

            if action == ord("d"):
                # delete the image
                cv2.destroyAllWindows()
                marked_for_deletion.append(img_path)
                marked_for_deletion.append(img_label)
                should_continue = True
            elif action == ord("n"):
                # skip the image
                should_continue = True
    
    # delete the images
    for f in marked_for_deletion:
        os.remove(f)


# find the labels that have no images
def find_missing_labels(path_to_images):
    train_images = [os.path.join(path_to_images, "images", "train", f) for f in os.listdir(os.path.join(path_to_images, "images", "train")) if f.endswith(".png")]
    val_images = [os.path.join(path_to_images, "images", "val", f) for f in os.listdir(os.path.join(path_to_images, "images", "val")) if f.endswith(".png")]
    train_labels = [os.path.join(path_to_images, "labels", "train", f) for f in os.listdir(os.path.join(path_to_images, "labels", "train")) if f.endswith(".txt")]
    val_labels = [os.path.join(path_to_images, "labels", "val", f) for f in os.listdir(os.path.join(path_to_images, "labels", "val")) if f.endswith(".txt")]

    missing_labels = []

    # get the img name and label name
    for train_label in train_labels:
        img_name = train_label.replace("labels", "images").replace(".txt", ".png")
        if img_name not in train_images:
            missing_labels.append(train_label)

inspect_dataset(output_dir)