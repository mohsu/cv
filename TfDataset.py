# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: dataset.py 
@time: 2020/06/02
"""

# python packages
from absl.flags import FLAGS

# 3rd-party packages
import imgaug as ia
import numpy as np
import tensorflow as tf
from loguru import logger

# self-defined packages
from utils.flag_utils import get_flag

IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/path': tf.io.FixedLenFeature([], tf.string),
    # 'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    # 'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def get_gen(dataset, batch_size):
    _iter = iter(dataset)
    while True:
        batch_x, batch_y = [], []
        for i in range(batch_size):
            x, y = next(_iter)
            batch_x.append(x)
            batch_y.append(y)
        yield batch_x, batch_y


@logger.catch(reraise=True)
class TfDataset:
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def __init__(self, size, num_channel, img_aug, num_classes, max_box=10, image_feature_map=None, label_smoothing_alpha=0.0):
        self.img_aug = img_aug
        self.size = size
        self.num_channel = num_channel
        self.num_classes = num_classes
        self.sum_num_classes = sum(num_classes)
        self.max_box = max_box
        self.image_feature_map = image_feature_map if image_feature_map is not None else IMAGE_FEATURE_MAP
        self._dataset = None
        self.label_smoothing_alpha = label_smoothing_alpha

    def dataset_map(self, map_func):
        self.dataset = self.dataset.map(map_func)

    def dataset_shuffle(self, buffer_size):
        self.dataset = self.dataset.shuffle(buffer_size)

    def dataset_batch(self, batch_size):
        self.dataset = self.dataset.batch(batch_size)
        # if self.org_dataset is not None:
        #     self.org_dataset = get_gen(self.org_dataset, batch_size)

    def prefetch_autotune(self):
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def build_example(annotation, process_label_to_index):
        width = int(annotation.size[1])
        height = int(annotation.size[0])

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        # views = []
        difficult_obj = []

        for obj in annotation.objects:
            difficult = bool(int(obj.difficult))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj.bbox.xmin))
            ymin.append(float(obj.bbox.ymin))
            xmax.append(float(obj.bbox.xmax))
            ymax.append(float(obj.bbox.ymax))
            # classes_text.append(str(obj.name).encode('utf8'))
            classes.append(process_label_to_index(obj.name))
            truncated.append(int(obj.truncated))
            # views.append(obj['pose'].encode('utf8'))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                annotation.annotation_path.encode('utf8')])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                annotation.annotation_path.encode('utf8')])),
            # 'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
            'image/path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation.image_path.encode('utf8')])),
            # 'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            # 'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['png'.encode('utf8')])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
            # 'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array(classes).ravel())),
            'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
            'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
            # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
        }))
        return example

    def load_voc_dataset(self, voc_set, process_label_to_index):
        examples = [self.build_example(annotation, process_label_to_index).SerializeToString() for annotation in
                    voc_set.annotations]
        dataset = tf.data.Dataset.from_tensor_slices(examples)
        return dataset

    @staticmethod
    def parse_detection_example(image_feature_map, num_category, num_channels):
        def _parse(example_s):
            features = tf.io.parse_example(example_s, image_feature_map)

            # x_train
            x_train = tf.io.read_file(features["image/path"])
            x_train = tf.image.decode_png(x_train, channels=num_channels)

            # y_train
            labels = tf.reshape(tf.cast(tf.sparse.to_dense(features["image/object/class/label"]), tf.float32),
                                (-1, num_category))
            y_train = tf.stack([tf.sparse.to_dense(features['image/object/bbox/xmin']),
                                tf.sparse.to_dense(features['image/object/bbox/ymin']),
                                tf.sparse.to_dense(features['image/object/bbox/xmax']),
                                tf.sparse.to_dense(features['image/object/bbox/ymax']),
                                ], axis=1)
            y_train = tf.concat([y_train, labels], axis=1)
            return x_train, y_train

        return _parse

    # @staticmethod
    # def parse_classification_example(example_s, image_feature_map):
    #     features = tf.io.parse_example(example_s, image_feature_map)
    #
    #     # x_train
    #     x_train = tf.io.read_file(features["image/path"])
    #     x_train = tf.image.decode_png(x_train, channels=get_flag("channels", 3))
    #     xmin, ymin, xmax, ymax = tf.sparse.to_dense(features['image/object/bbox/xmin']), \
    #                              tf.sparse.to_dense(features['image/object/bbox/ymin']), \
    #                              tf.sparse.to_dense(features['image/object/bbox/xmax']), \
    #                              tf.sparse.to_dense(features['image/object/bbox/ymax'])
    #     x_train = x_train[ymin:ymax + 1, xmin: xmax + 1, :]
    #     y_train = tf.cast(tf.sparse.to_dense(features["image/object/class/label"]), tf.float32)
    #     return x_train, y_train

    def augment_data(self, x_train, y_train):
        image = x_train

        bbs = []
        for box in y_train:
            bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=box[4:]))
        bboi = ia.BoundingBoxesOnImage(bbs, shape=image.shape)

        images, bboxes = self.img_aug.aug([image], [bboi])

        y_train = np.zeros((get_flag("yolo_max_boxes", self.max_box), 4 + self.sum_num_classes), dtype=np.float32)
        bbs = bboxes[0].remove_out_of_image().clip_out_of_image()

        for i, bbox in enumerate(bbs.bounding_boxes):
            # skip any coord < 0 or >= size bbox after augmentation
            if not 0 < bbox.center_x < self.size or not 0 < bbox.center_y < self.size:
                continue
            y_train[i][0:4] = bbox.x1 / self.size, bbox.y1 / self.size, bbox.x2 / self.size, bbox.y2 / self.size
            # convert label to one hot
            y_idx_start = 4
            for idx_cat in range(len(self.num_classes)):
                y_idx_end = y_idx_start + self.num_classes[idx_cat]
                y_train[i][y_idx_start:y_idx_end] = np.eye(self.num_classes[idx_cat])[int(bbox.label[idx_cat])]
                y_idx_start = y_idx_end

        x_train = tf.convert_to_tensor(images[0], dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        return x_train, y_train

    @tf.function
    def tf_augment_data(self, x_train, y_train):
        x_train, y_train = tf.numpy_function(self.augment_data, [x_train, y_train], [tf.float32, tf.float32])

        x_train.set_shape((self.size, self.size, self.num_channel))
        y_train.set_shape((get_flag("yolo_max_boxes", self.max_box), 4 + self.sum_num_classes))

        return x_train, y_train

    def load_image_dataset(self, image_paths):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        return dataset

    def parse_image(self, filename):
        # x_train
        x_train = tf.io.read_file(filename)
        x_train = tf.image.decode_png(x_train, channels=self.num_channel)
        return x_train

    def augment_image(self, x_train):
        image = x_train
        images, _ = self.img_aug.aug([image])
        x_train = tf.convert_to_tensor(images[0], dtype=tf.float32)
        return x_train

    @tf.function
    def tf_augment_image(self, x_train):
        [x_train] = tf.numpy_function(self.augment_image, [x_train], [tf.float32])
        x_train.set_shape((self.size, self.size, self.channels))

        return x_train

    @staticmethod
    @tf.function
    def transform_targets_for_output(y_true, grid_size, anchor_idxs, sum_num_classes):
        # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
        N = tf.shape(y_true)[0]

        # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
        y_true_out = tf.zeros(
            (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 4 + 1 + sum_num_classes))

        anchor_idxs = tf.cast(anchor_idxs, tf.int32)

        indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        idx = 0

        for i in tf.range(N):
            for j in tf.range(tf.shape(y_true)[1]):
                if tf.equal(y_true[i][j][2], 0):
                    continue
                anchor_eq = tf.equal(
                    anchor_idxs, tf.cast(y_true[i][j][-1], tf.int32))  # best anchor

                if tf.reduce_any(anchor_eq):
                    box = y_true[i][j][0:4]
                    box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                    # tf.print(box)

                    anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                    grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                    # grid[y][x][anchor] = (tx, ty, bw, bh, obj(1), class)
                    # tf.print("i", grid_xy[0], "j", grid_xy[1], "k", anchor_idx[0][0], "c", tf.argmax(y_true[i][j][4:-1]))
                    indexes = indexes.write(
                        idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                    value = tf.concat([box, [1], y_true[i][j][4:-1]], axis=0)
                    updates = updates.write(idx, value)
                    idx += 1

        if not tf.equal(idx, 0):
            # tf.print(indexes.stack())
            # tf.print(updates.stack())
            return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())
        else:
            return y_true_out

    def transform_targets(self, y_train, grid_factor, anchors, anchor_masks):
        y_outs = []
        grid_size = self.size // grid_factor

        # calculate anchor index for true boxes
        anchors = tf.cast(anchors, tf.float32)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                         (1, 1, tf.shape(anchors)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                       tf.minimum(box_wh[..., 1], anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

        y_train = tf.concat([y_train, anchor_idx], axis=-1)

        for anchor_idxs in anchor_masks:
            y_outs.append(self.transform_targets_for_output(y_train, grid_size, anchor_idxs, self.sum_num_classes))
            grid_size *= 2

        return tuple(y_outs)
