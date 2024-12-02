import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class TableResize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, results):
        self.resize_img(results)
        self.resize_box(results)
        return results

    def resize_img(self, results):
        image = results["img"]
        h, w, _ = image.shape
        if w < h:
            w = int(self.size / h * w)
            h = int(self.size)
        else:
            h = int(self.size / w * h)
            w = int(self.size)
        scale = (h / image.shape[0], w / image.shape[1])
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        results.update(img=image, img_shape=image.shape, img_scale=scale)

    def resize_box(self, results):
        h, w = results["img_shape"][:2]
        info = results.get("img_info")
        # train and val phase
        if info is not None and info.get("bbox") is not None:
            bbox = info["bbox"]
            y, x = results["img_scale"]
            bbox[..., 0::2] = np.clip(bbox[..., 0::2] * x, 0, w - 1)
            bbox[..., 1::2] = np.clip(bbox[..., 1::2] * y, 0, h - 1)
            info.update(bbox=bbox)


@PIPELINES.register_module()
class TablePad:
    def __init__(self, size):
        self.size = size[::-1]

    def __call__(self, results):
        img = self.extend(results["img"], self.size)
        results.update(img=img, pad_shape=img.shape)
        return results

    def extend(self, img, size):
        h = (0, size[0] - img.shape[0])
        w = (0, size[1] - img.shape[1])
        pad = np.pad(img, (h, w, (0, 0)))
        return pad.astype(img.dtype)


@PIPELINES.register_module()
class TableBboxFlip:
    def __call__(self, results):
        h, _, _ = results["img_shape"]
        bbox = results["img_info"].get("bbox", results.get("bbox"))
        mask = np.count_nonzero(bbox, axis=-1, keepdims=True)
        flip = np.where(mask, h - 1 - bbox, bbox).clip(min=0)
        np.copyto(bbox[..., 1], flip[..., 1])
        np.copyto(bbox[..., 3], flip[..., 3])
        return results


@PIPELINES.register_module()
class TableBboxEncode:
    def __call__(self, results):
        info = results["img_info"]
        size = results["img"].shape
        bbox = self.xyxy_to_xywh(info["bbox"])
        bbox = self.normalize_bbox(bbox, size)
        assert np.all(bbox >= 0)
        assert np.all(bbox <= 1)
        info.update(bbox=bbox)
        self.adjust_key(results)
        return results

    def xyxy_to_xywh(self, bbox):
        bb = np.empty_like(bbox)
        # xy center
        bb[..., 0] = bbox[..., 0::2].mean(axis=-1)
        bb[..., 1] = bbox[..., 1::2].mean(axis=-1)
        # width and height
        bb[..., 2] = bbox[..., 0::2].ptp(axis=-1)
        bb[..., 3] = bbox[..., 1::2].ptp(axis=-1)
        return bb

    def normalize_bbox(self, bbox, size):
        bbox[..., 0::2] /= size[1]
        bbox[..., 1::2] /= size[0]
        return bbox

    def adjust_key(self, results):
        results.update(html=results["img_info"].pop("html"))
        results.update(cell=results["img_info"].pop("cell"))
        results.update(bbox=results["img_info"].pop("bbox"))
