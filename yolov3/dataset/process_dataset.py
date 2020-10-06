# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: process_dataset.py
@time: 2020/06/18
"""

# python packages

# 3rd-party packages
from loguru import logger

# self-defined packages
from cv.yolov3.core.models import get_anchor_masks
from cv.TfDataset import TfDataset
from utils.flag_utils import get_flag


def process_train_dataset(model, train_p, batch_size):
    train_p.dataset_map(train_p.parse_detection_example(train_p.image_feature_map, len(model.num_classes), model.channels))
    train_p.dataset_map(train_p.tf_augment_data)
    train_p.dataset_shuffle(get_flag("batch_size", batch_size))
    train_p.dataset_batch(get_flag("batch_size", batch_size))
    train_p.dataset_map(
        lambda x, y: (x,
                      train_p.transform_targets(y,
                                                model.grid_factor,
                                                model.anchors,
                                                get_anchor_masks(model.anchors))))


@logger.catch(reraise=True)
def get_train_dataset(model, voc_set, batch_size=24):
    """train/validation dataset"""
    train_p = TfDataset(model.input_size[0], model.channels, model.img_aug, model.num_classes, model.max_box)
    train_p.dataset = train_p.load_voc_dataset(voc_set.training, model.to_index)

    process_train_dataset(model, train_p, batch_size)
    train_p.prefetch_autotune()

    val_p = TfDataset(model.input_size[0], model.channels, model.img_aug, model.num_classes)
    val_p.dataset = train_p.load_voc_dataset(voc_set.validation, model.to_index)
    process_train_dataset(model, val_p, batch_size)

    return train_p.dataset, val_p.dataset


@logger.catch(reraise=True)
def get_test_dataset(model, voc_set, batch_size=24):
    test_p = TfDataset(model.input_size[0], model.channels, model.img_aug, model.num_classes, model.max_box)
    test_p.dataset = test_p.load_voc_dataset(voc_set.testing, model.to_index)
    test_p.dataset_map(test_p.parse_detection_example(test_p.image_feature_map, len(model.num_classes), model.channels))
    test_p.dataset_map(test_p.tf_augment_data)
    test_p.dataset_batch(get_flag("batch_size", batch_size))

    # org_p = TfDataset(model.input_size[0], model.img_aug, len(model.class_names))
    # org_p.dataset = org_p.load_voc_dataset(voc_set.testing, model.to_index)
    # org_p.dataset_map(org_p.parse_detection_example(org_p.image_feature_map))
    # org_p.dataset_batch(get_flag("batch_size", batch_size))

    return test_p.dataset
