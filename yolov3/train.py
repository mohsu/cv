# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: train.py 
@time: 2020/05/22

# FLAGS.flags_into_string()
# FLAGS.append_flags_into_file('path/to/flagfile.txt')
"""

# python packages
import datetime
from absl import flags, app
from absl.flags import FLAGS

# 3rd-party packages
import numpy as np
import tensorflow as tf
import pandas as pd
from loguru import logger
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,

)
from tensorflow.keras.experimental import (
    CosineDecay
)
# self-defined packages
from utils import os_path, image_processing, flag_utils
from cv.yolov3.core.layers import (
    YoloLoss
)
from cv.yolov3.core.models import (
    YoloV3, YoloV3Tiny
)
from cv.yolov3.core import utils as yolo_utils
from utils.keras_utils import LogHistory, PlotHistory

flags.DEFINE_integer("size", default=416, help="size of image")
flags.DEFINE_integer('epochs', default=2, help='number of epochs')
flags.DEFINE_integer('batch_size', default=24, help='batch size')
flags.DEFINE_integer('early_stop', default=5, help='early stopping')

flags.DEFINE_float('learning_rate', default=1e-3, help='learning rate')
"""
learning rate decay scheduler
flags.DEFINE_integer('lr_patience', default=20, help='lr_patience')
"""
flags.DEFINE_integer("decay_steps", default=50 , help="Number of steps to decay over.")
flags.DEFINE_float("lr_alpha", default=0.9, help="Minimum learning rate value as a fraction of initial_learning_rate")

flags.DEFINE_integer('freeze', default=0, help='freeze layers before xth layer')

flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', None, 'path to weights file')
flags.DEFINE_enum('fit', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'same',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune', 'same'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only, '
                  'same: Use previous trained model')
# flags.DEFINE_list('loss_weight', [1.0, 1.0, 1.0, 1.0], "[xy, wh, obj, class] loss weight")
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                                                  'useful in transfer learning with different number of classes')


def run_eager_fit(model, train_dataset, val_dataset, optimizer, loss):
    # Eager mode is great for debugging
    # Non eager graph mode is recommended for real training
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    for epoch in range(1, FLAGS.epochs + 1):
        for batch, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            logger.info(
                f"[Epoch{epoch}/{FLAGS.epochs}][Train][Batch {batch + 1}] "
                f"Loss: {total_loss.numpy()}, {list(map(lambda x: np.sum(x.numpy()), pred_loss))}")
            avg_loss.update_state(total_loss)

        for batch, (images, labels) in enumerate(val_dataset):
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            logger.info(
                f"[Epoch{epoch}/{FLAGS.epochs}][Val][Batch {batch}] "
                f"Loss: {total_loss.numpy()}, {list(map(lambda x: np.sum(x.numpy()), pred_loss))}")
            avg_val_loss.update_state(total_loss)

        logger.info(
            f"[Epoch {epoch}][Summary] train: {avg_loss.result().numpy()}, val: {avg_val_loss.result().numpy()}")
        model.save_weights(os_path.join(os_path.dirname(FLAGS.weights), "checkpoints",
                                        f'ep{epoch:03d}-loss{avg_loss.result().numpy():.3f}'
                                        f'-val_loss{avg_val_loss.result().numpy():.3f}.h5'), )
        avg_loss.reset_states()
        avg_val_loss.reset_states()


@logger.catch(reraise=True)
def train(model, train_dataset, val_dataset):

    flag_utils.log_flag()
    """Configure the model for transfer learning"""
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny()
        else:
            model_pretrained = YoloV3()
        model_pretrained.load_weights(FLAGS.weights, by_name=True)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            yolo_utils.freeze_all(model.get_layer('yolo_darknet'), until_layer=FLAGS.freeze)

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    yolo_utils.freeze_all(l, until_layer=FLAGS.freeze)

    else:
        # All other transfer require matching classes
        model.load_weights(weight_path=FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            yolo_utils.freeze_all(darknet, until_layer=FLAGS.freeze)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            yolo_utils.freeze_all(model, until_layer=FLAGS.freeze)

    lr_scheduler = CosineDecay(initial_learning_rate=FLAGS.learning_rate, decay_steps=FLAGS.decay_steps,
                               alpha=FLAGS.lr_alpha)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss = [YoloLoss(model.anchors[mask], classes=model.num_classes) for mask in
            model.anchor_masks]

    """Start training"""
    if FLAGS.fit == 'eager_tf':
        run_eager_fit(model, train_dataset, val_dataset, optimizer, loss)
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))
        checkpoint_path = os_path.join(os_path.dirname(FLAGS.weights), "checkpoints",
                                       'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
        os_path.make_dir(os_path.dirname(checkpoint_path))
        plt_his = PlotHistory()
        callbacks = [
            # ReduceLROnPlateau(verbose=1, patience=FLAGS.lr_patience),
            # CosineDecay(initial_learning_rate=FLAGS.learning_rate)
            EarlyStopping(patience=FLAGS.early_stop, verbose=1),
            ModelCheckpoint(checkpoint_path,
                            monitor='val_loss', verbose=0, save_best_only=False,
                            save_weights_only=True, mode='auto', save_freq='epoch'),
            # TensorBoard(log_dir='../logs'),
            LogHistory(),
            plt_his
        ]
        try:
            history = model.fit(train_dataset,
                                epochs=FLAGS.epochs,
                                callbacks=callbacks,
                                validation_data=val_dataset)
        except Exception as e:
            raise e
        finally:
            plt_his.plot(save_path=os_path.join(os_path.dirname(FLAGS.weights),
                                                f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.png"))


@logger.catch(reraise=True)
def test(model, test_dataset, voc_set, box_thres=15):
    model.load_weights(weight_path=FLAGS.weights)

    thres = box_thres
    box_col = ["x1", "y1", "x2", "y2"]
    class_names = [x.__name__ for x in model.class_names]
    true_class_names = [f"{x}_true" for x in class_names]

    # get all data
    test_image_shape = []
    test_labels = []
    for batch_test_images, batch_test_labels in test_dataset.as_numpy_iterator():
        test_image_shape = batch_test_images.shape[1:]
        test_labels.append(batch_test_labels)
    test_labels = np.concatenate(test_labels, axis=0)

    # predict on dataset
    from utils.timeit import Timer
    # t = Timer()
    preds = model.predict(test_dataset)
    # pred_boxes, scores = yolo_utils.nms(preds, test_image_shape=test_image_shape)
    pred_boxes, scores = yolo_utils.decode_tf_nms(preds, test_image_shape)

    pred_labels = yolo_utils.convert2_class(scores, model.num_classes)

    true_boxes = list(yolo_utils.rescale_pred_wh(test_labels[..., :4], test_image_shape))
    true_boxes = [yolo_utils.trim_zeros_2d(x) for x in true_boxes]

    true_labels = yolo_utils.convert2_class(test_labels[..., 4:], model.num_classes)
    true_labels = [true_labels[i][0][:len(true_boxes[i])] for i in range(len(true_labels))]

    df_preds, df_trues = pd.DataFrame(), pd.DataFrame()
    for i in range(len(pred_boxes)):  # concat by sample
        df_pred_box = pd.DataFrame.from_records(pred_boxes[i], columns=box_col)
        df_true_box = pd.DataFrame.from_records(true_boxes[i], columns=box_col)

        df_pred_label = pd.DataFrame.from_records(pred_labels[i][0], columns=class_names)
        df_true_label = pd.DataFrame.from_records(true_labels[i], columns=class_names)
        df_pred_score = pd.DataFrame.from_records(pred_labels[i][1], columns=[f"{x}_score" for x in class_names])

        df_pred = pd.concat([df_pred_box, df_pred_score, df_pred_label], axis=1)
        df_pred["index"] = i
        df_pred["obj"] = range(len(df_pred))

        df_true = pd.concat([df_true_box, df_true_label], axis=1)
        df_true["index"] = i
        df_true["obj"] = range(len(df_true))

        df_preds = df_preds.append(df_pred, ignore_index=True)
        df_trues = df_trues.append(df_true, ignore_index=True)

    total_count = len(df_preds)
    total_label_count = total_count * len(class_names)

    df_preds = df_preds.groupby(by="index").apply(lambda x:
                                                  x.sort_values(["x1", "y1"])).reset_index(drop=True)

    # # add obj_index
    # df_preds = df_preds.groupby(by="index").apply(lambda x:
    #                                               pd.concat([x.reset_index(drop=True),
    #                                                          pd.DataFrame({"obj": range(len(x))})],
    #                                                         axis=1)).reset_index(drop=True)
    # df_trues = df_trues.groupby(by="index").apply(lambda x:
    #                                               pd.concat([x.reset_index(drop=True),
    #                                                          pd.DataFrame({"obj": range(len(x))})],
    #                                                         axis=1)).reset_index(drop=True)

    # expand to size of each other
    df_trues_can_repeat = df_trues[df_trues["index"].isin(df_preds["index"].unique())]
    df_preds_repeat = df_preds.groupby(by="index").apply(
        lambda x: pd.concat([x] * len(df_trues[df_trues["index"] == x["index"].unique()[0]]))).reset_index(drop=True)
    df_trues_repeat = df_trues_can_repeat.groupby(by="index").apply(
        lambda x: pd.concat([x] * len(df_preds[df_preds["index"] == x["index"].unique()[0]]))).reset_index(drop=True)

    # reset indexes

    df_trues_repeat = df_trues_repeat.groupby(by="index").apply(lambda x: x.sort_values("obj")).reset_index(drop=True)

    df_preds_repeat["diff"] = df_preds_repeat[box_col].subtract(df_trues_repeat[box_col]).abs().sum(axis=1)
    df_preds_repeat["true_idx"] = df_trues_repeat["obj"]

    df_preds = df_preds_repeat.loc[df_preds_repeat.groupby(["index", "obj"])["diff"].idxmin()]

    df_all = pd.merge(df_preds, df_trues, how="left", left_on=["true_idx", "index"], right_on=["obj", "index"],
                      suffixes=("", "_true")).reset_index(drop=True)

    # box_diff are huge (false positive)
    df_all["w"] = df_all["x2_true"] - df_all["x1_true"]
    df_all["h"] = df_all["y2_true"] - df_all["y1_true"]
    df_all["diff_rate"] = df_all["diff"] / (df_all["w"] + df_all["h"]) / 2
    false_positive_idx = df_all[(df_all["diff_rate"] > thres) & (~df_all.isna().any(axis=1))].index
    df_all_1 = df_all.copy()

    # label mismatch
    df_all = df_all.loc[~df_all.index.isin(false_positive_idx)]
    is_label_mismatch = df_all[class_names] != df_all[true_class_names].values
    label_mismatch_idx = df_all[is_label_mismatch.any(axis=1)].index

    # miss prediction (true negative)
    true_obj_counts = df_trues.groupby(by="index").size().reset_index(name="true_count")
    df_all = pd.merge(df_all, true_obj_counts, on="index")
    miss_label_index = df_all.groupby(by="index").apply(lambda x: np.setdiff1d(np.arange(x["true_count"][0]), x["obj"]))
    miss_label_index = miss_label_index[miss_label_index.apply(lambda x: len(x) > 0)]
    if len(miss_label_index) > 0:
        df_missed = pd.DataFrame(miss_label_index, columns=["obj"]).reset_index("index").explode("obj")
        df_true_label_misses = pd.merge(df_missed, df_trues, how="left")
    else:
        df_true_label_misses = pd.DataFrame()
    df_all = df_all_1

    true_negative_count = len(df_true_label_misses)
    total_box_loss = df_all["diff"].sum()
    total_false_positive_count = len(false_positive_idx)
    class_diff = {class_name: is_label_mismatch[class_name].sum() for class_name in class_names}

    logger.debug(
        f"Total miss detection count={true_negative_count}, avg tn={true_negative_count / total_count:.4f}")
    logger.debug(f"Total box loss={total_box_loss}, avg box loss={total_box_loss / total_count:.4f}")
    logger.debug(
        f"Total false positive count={total_false_positive_count}, avg fp={total_false_positive_count / total_count:.4f}")
    logger.debug(
        f"Total misclassification: {sum(class_diff.values())}, classes diff: {class_diff} "
        f"class acc: {1 - sum(class_diff.values()) / total_label_count}")
    logger.debug(
        f"Total accuracy= {1 - (sum(class_diff.values()) + total_false_positive_count + true_negative_count) / total_count}")

    def get_bbox(df_bbox):
        return list(df_bbox.values.ravel().astype('int'))

    def get_label(classes):
        if isinstance(classes, str):
            return classes
        if not classes.empty:
            classes = classes.values.ravel()
            return "".join([str(model.class_names[i].get_pair(int(classes[i]))) for i in range(len(class_names))])
        else:
            return ""

    def display(image_path, bbox, pred_classes, true_classes, title):
        image_org = image_processing.imread(image_path)
        image = model.img_aug.aug([image_org])[0][0]
        image = image_processing.draw_box(image, bbox, label=get_label(pred_classes))
        image_path = os_path.join(os_path.dirname(image_path, depth=2, full_path=False),
                                  os_path.get_filename(image_path))
        image_processing.show(image, title=f"{title} true={get_label(true_classes)}\n {image_path}")

    # display
    for idxes, title in [[false_positive_idx, "False Positive"], [label_mismatch_idx, "Mismatch"]]:
        for i in idxes:
            row = df_all.loc[i]
            dic_row = row.to_dict()
            dic_row["image_path"] = voc_set.image_paths[int(row["index"])]
            print(dic_row)
            display(voc_set.image_paths[int(row["index"])], get_bbox(row[box_col]), row[class_names],
                    row[true_class_names],
                    title)

    for i, row in df_true_label_misses.iterrows():
        display(voc_set.image_paths[int(row["index"])], get_bbox(row[box_col]), "XX", row[class_names], "True Negative")
