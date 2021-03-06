# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: model.py 
@time: 2020/06/01
"""

# python packages
import socket
from collections import defaultdict

# 3rd-party packages
import grpc
import imgaug as ia
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from tensorflow.keras.models import load_model, Model
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2

# self-defined packages
from cv.yolov3.core.models import get_anchor_masks
from cv.yolov3.core.models import (
    YoloCustomizeModelMultiCategory
)
from cv.yolov3.core.utils import (
    nms,
    decode_tf_nms,
    convert2_class
)
from utils import os_path, image_processing
from utils.label.VOCLabel import (
    VOCAnnotation,
    VOCObject
)
from utils.timeit import Timer


@logger.catch(reraise=True)
class ImageAug:
    def __init__(self, aug_seq=None, resize_seq=None):
        self.aug_seq = aug_seq
        self.resize_seq = resize_seq

    def aug(self, images, labels=None):
        seqs = []
        if self.aug_seq:
            seqs.append(self.aug_seq)

        if self.resize_seq:
            seqs.append(self.resize_seq)

        for seq in seqs:
            seq = seq.to_deterministic()
            if not labels:
                images = seq.augment_images(images)
            else:
                images, labels = seq(images=images, bounding_boxes=labels)
        return images, labels


class TFServeModel(Model):
    def __init__(self, model_dir, version=None):
        super(TFServeModel, self).__init__()
        self.model_dir = model_dir
        if version is None:
            version = os_path.list_dir(model_dir, sort=True, sort_key=os_path.getctime, full_path=False)[-1]
        self.version = version
        model = tf.saved_model.load(os_path.join(self.model_dir, self.version))
        self.__dict__.update(model.__dict__)

    def predict(self, *args, **kwargs):
        return self.signatures["serving_default"](*args, **kwargs)


class CNNModel(Model):
    def __init__(self, model_name, input_size):
        super(CNNModel, self).__init__()
        input_size = tuple(map(int, input_size))
        self.input_size = input_size
        self.model_name = model_name

    def load(self, model_dir=None, model_path=None):
        if model_dir is None and model_path is None:
            raise Exception("No model path can be loaded")
        if model_path is None:
            model_path = os_path.join(model_dir, f"{self.model_name}.h5")
        if not os_path.exists(model_path):
            logger.warning("Pretrained model does not exist in {}.".format(model_path))
            return
        logger.debug('Load model {}.'.format(model_path))
        model = load_model(model_path, compile=False)
        self.__dict__.update(model.__dict__)

    def load_weights(self, weight_dir=None, weight_path=None, force=False, **kwargs):
        if not force:
            if weight_dir is None and weight_path is None:
                raise Exception("No model_weight path can be loaded")
            if weight_path is None:
                weight_path = os_path.join(weight_dir, f"{self.model_name}.h5")
            if not os_path.exists(weight_path):
                logger.warning("Pretrained weights does not exist in {}.".format(weight_path))
                return

        logger.debug('Load weights {}.'.format(weight_path))
        super(CNNModel, self).load_weights(weight_path, **kwargs)

    def save_to_serve(self, model_dir, version="0001", overwrite=True):
        output_dir = os_path.join(model_dir, self.model_name, version)
        tf.saved_model.save(self, output_dir)
        logger.debug(f"model has been exported to {output_dir}")


class ClassificationModel(CNNModel):
    def __init__(self, name, input_size, list_class_names):
        super().__init__(name, input_size)
        self.list_class_names = list_class_names

    def get_pred_df(self, *args, **kwargs):
        raise NotImplementedError

    def predicted_to_klasses(self, pred):
        """
        :param pred: scores of each classes [np.array(n_sample), np.array(n_sample)... len(list_class_names)]
        :return: [[klass_1, klass_2], [klass_1, klass_2], ...n_sample]
        """
        for n, klass_name in enumerate(self.list_class_names):
            pred[n] = np.argmax(pred[n], axis=-1)
        klasses = []
        for i in range(len(pred[0])):
            klasses_of_one_sample = []
            for n, klass_name in enumerate(self.list_class_names):
                klasses_of_one_sample.append(klass_name[pred[n][i]])
            klasses.append(klasses_of_one_sample)
        return klasses

    def test(self, gen_test, output_dir=None):
        if self.model is None:
            raise Exception("Model is not defined.")

        images, labels = gen_test.get_all()

        logger.debug("Predicting... ")

        pred = self.predict(images)

        df_pred = self.get_pred_df(pred)

        df_ans = pd.DataFrame(labels)
        df_ans = df_ans.applymap(lambda x: np.argmax(x))

        df_right = (df_pred == df_ans)

        for num_klass, klass_name in enumerate(df_right):
            logger.debug("Label {} acc: {}/{} = {:.4f}".format(klass_name, np.sum(df_right[klass_name]), len(df_right),
                                                               np.sum(df_right[klass_name]) / len(df_right)))

            # save wrong prediction
            for i in df_pred[klass_name].loc[df_ans[klass_name] != df_pred[klass_name]].index:
                annotation = gen_test.dataset.annotations[i]
                title = f"ans({self.list_class_names[num_klass][df_ans[i]]})_pred({self.list_class_names[num_klass][df_pred[i]]})"

                image = image_processing.draw_annotation(annotation)
                if output_dir is None:
                    image_processing.show(image, title=f"{annotation.filename}\n{title}")
                else:
                    image_processing.save_image(image, os_path.join(output_dir, f"{annotation.filename}_{title}.png"))

        logger.debug(
            "Total acc {}/{} = {:.4f}".format(np.sum(df_right.all(axis=1)), len(df_right),
                                              np.sum(df_right.all(axis=1)) / len(df_right)))


class YoloModel(CNNModel):
    @property
    def num_classes(self):
        return [len(klass) for klass in self.class_names]

    def to_named_classes(self, indexes):
        named_classes = [self.class_names[i].get_pair(indexes[i]) for i in range(len(indexes))]
        if len(named_classes) == 1:
            return named_classes[0]
        return named_classes

    def __init__(self, name, input_size, class_names,
                 training, base_model, anchors, channels=3, grid_factor=32,
                 normalize_input=True, expand_pixel=0, min_box_size=None, connected=False, max_box=10):
        super().__init__(name, input_size)
        self.class_names = class_names
        self.training = training
        self.anchors = anchors
        self.anchor_masks = get_anchor_masks(anchors)
        self.channels = channels
        self.img_aug = None
        self.grid_factor = grid_factor
        self.expand_pixel = expand_pixel
        self.max_box = max_box

        yolo_model = YoloCustomizeModelMultiCategory(input_size[0],
                                                     channels=channels,
                                                     anchors=anchors,
                                                     classes=self.num_classes,
                                                     training=training,
                                                     model=base_model,
                                                     normalize_input=normalize_input,
                                                     min_box_size=min_box_size,
                                                     connected=connected,
                                                     max_box=max_box)

        self.__dict__.update(yolo_model.__dict__)
        self._grpc_channels = defaultdict()

    def process_test_data(self, images, convert_BGR=False):
        images = self.img_aug.aug(images)[0]
        if self.channels != 3:
            if convert_BGR:
                cvt_code = image_processing.COLOR_RGB2GRAY
            else:
                cvt_code = image_processing.COLOR_BGR2GRAY
            images = [np.expand_dims(image_processing.cvtColor(image, cvt_code), -1) for image in images]
        images = np.array(images)
        if self.channels == 3 and convert_BGR:
            blue = images[..., 0].copy()
            images[..., 0], images[..., 2] = images[..., 2], blue

        return images

    def get_grpc_channel(self, ip="0.0.0.0", port=8500, use_dns=False, reopen=False):
        if use_dns:
            ip = socket.gethostbyname(str(ip))
        server = f"{ip}:{port}"

        def log_connectivity_changes(connectivity):
            logger.debug("Channel changed status to %s." % connectivity)

        def restart_a_channel(server):
            logger.debug(f"Restarting grpc channel {server}.")
            channel = grpc.insecure_channel(server)
            # channel.subscribe(log_connectivity_changes)
            return channel

        channel = self._grpc_channels.get(server)
        if channel is None or reopen:
            channel = restart_a_channel(server)
        else:
            # check connectivity state
            connectivity_state = channel._connectivity_state.channel.check_connectivity_state(True)
            if connectivity_state == grpc.ChannelConnectivity.SHUTDOWN:
                # if it's dead restart then restart a channel
                channel = restart_a_channel(server)

        # assign to dict
        self._grpc_channels[server] = channel

        return self._grpc_channels[server]

    def predict_on_docker(self, test_images, version="0001", ip="0.0.0.0", port=9000, use_dns=False):
        stub = prediction_service_pb2_grpc.PredictionServiceStub(self.get_grpc_channel(ip, port, use_dns))
        # create request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = 'serving_default'
        tensor = tf.make_tensor_proto(test_images, shape=test_images.shape, dtype=types_pb2.DT_FLOAT)
        request.inputs['input'].CopyFrom(tensor)
        try:
            r = stub.Predict.future(request, 2.5)  # 5 seconds
            results = r.result()
        except grpc.RpcError as rpc_error:
            if rpc_error.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN]:
                logger.exception(rpc_error)
                self.get_grpc_channel(ip, port, use_dns, reopen=True)
            raise rpc_error
        preds = [tf.make_ndarray(results.outputs[key]) for key in self.output_names]
        # data = json.dumps({"signature_name": "serving_default", "instances": test_images.tolist()})
        #
        # headers = {"content-type": "application/json"}
        # json_response = requests.post(f'http://localhost:{port}/v1/models/{self.model_name}:predict', data=data,
        #                               headers=headers)
        # r = json.loads(json_response.text)
        # if "predictions" in r:
        #     predictions = r['predictions']
        #     preds = defaultdict(list)
        #     for i in range(len(test_images)):
        #         for key in predictions[i].keys():
        #             preds[key].append(tf.convert_to_tensor(predictions[i][key]))
        #     preds = [tf.convert_to_tensor(preds[key]) for key in sorted(predictions[0].keys())]
        # else:
        #     preds = []
        return preds

    def detect(self, images, convert_BGR=False, to_annotation=False, on_port=None):
        """
        preprocess -> predict -> postprocess
        :param on_port:
        :param images: list of images that has the same shape
        :param convert_BGR: if true, will convert BGR in preprocess
        :param to_annotation: if true, will convert to VOCAnnotation
        :return:
        """
        if not isinstance(images, list) and (isinstance(images, np.ndarray) and images.ndim == 3):
            images = [images]

        reverse_seq = self.get_reverse_seq(images[0].shape[:2])

        processed_images = self.process_test_data(images, convert_BGR)
        if on_port:
            timer = Timer(f"[{os_path.os.getpid()}][{self.model_name}] predict on docker")
            preds = self.predict_on_docker(processed_images, port=on_port)
            timer.stop()
        else:
            preds = self.predict(processed_images)

        pred_boxes, scores = decode_tf_nms(preds, self.input_size + (self.channels,))
        pred_labels = convert2_class(scores, self.num_classes)

        # rescale bounding boxes
        def rescale_bbox(pred_box):
            bbox = ia.BoundingBoxesOnImage.from_xyxy_array(pred_box, shape=self.input_size + (self.channels,))
            reversed_bbox = reverse_seq(bounding_boxes=bbox).to_xyxy_array().astype('int32')
            return reversed_bbox

        pred_boxes = [rescale_bbox(pred_box) for pred_box in pred_boxes]
        if not to_annotation:
            return pred_boxes, pred_labels

        annotations = []
        for n_image in range(len(images)):
            pred_klass, pred_score = pred_labels[n_image]
            objs = []

            for n_obj in range(len(pred_klass)):
                # process labels
                named_classes = self.to_named_classes(pred_klass[n_obj])

                # process score
                pred_score_obj = pred_score[n_obj]
                if len(pred_score_obj) == 1:
                    pred_score_obj = pred_score_obj[0]

                # create VOCObject
                objs.append(VOCObject(name=named_classes, bbox=list(pred_boxes[n_image][n_obj]), score=pred_score_obj))

            annotations.append(VOCAnnotation(size=images[n_image].shape[:2], objects=objs))
        return annotations

    def detect_helper(self, *args, **kwargs):
        queue = args[-1]
        annotation = self.detect(*args[:-1], **kwargs)
        queue.put(annotation)
