from ..builder import DETECTORS
from .single_stage import SingleStageDetector

import mmcv
import numpy as np
import torch
from mmdet.core import bbox2result

import cv2
import numpy as np

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

@DETECTORS.register_module()
class GFL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
                #  demo=False):
        super(GFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def simple_test(self, img, img_metas, rescale=False, demo=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        # return bbox_results
        # score_list=[]
        # for cls_score in outs[0]:
        #     scores = cls_score.sigmoid()
        #     score_list.append(scores)

        # uncertainty_list = outs[3]
        # uncertainty_return_list = []
        # for score, uncert in zip(score_list, uncertainty_list):
        #     max_score, max_idx = score.max(dim=1)
        #     # uncert_cls = torch.sqrt((1.0 - uncert) * max_score * 10)
        #     uncert_cls= torch.stack([max_score, max_score, max_score, max_score], dim=0)
        #     uncertainty_return_list.append(uncert_cls)

        
        if(demo) : return bbox_results[0], outs[2], outs[3]
        # if(self.DEMO) : return bbox_results[0], uncertainty_return_list
        else : return bbox_results

   

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
   
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 9
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 9
        scores = bboxes[:, 4]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]


    text_color = color_val(text_color)
    # color_1 = color_val('magenta')
    # color_2 = color_val('blue')
    # color_3 = color_val('green')


    img = np.ascontiguousarray(img)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])

        # if(i % 2 == 0) : bbox_color = color_val(bbox_color)
        #else : bbox_color = (0, 127, 255)

        bbox_color = color_val(bbox_color)
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)

        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        
        uncertainty_left_text = f'{bbox[5]:.02f}'
        uncertainty_top_text = f'{bbox[6]:.02f}'
        uncertainty_right_text = f'{bbox[7]:.02f}'
        uncertainty_bottom_text = f'{bbox[8]:.02f}'

        bbox_width = (bbox_int[2] - bbox_int[0])//2
        bbox_height = (bbox_int[3] - bbox_int[1])//2

        font = cv2.FONT_HERSHEY_PLAIN
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # set the rectangle background to white
        rectangle_bgr_fill = (0, 0, 0)
        rectangle_brg_line = text_color

        
        if len(bbox) > 4:
            label_text += f'|{bbox[4]:.02f}'
        
        # label_text += uncertainty_top_text

        # get the width and height of the text box
        txt = label_text
        (text_width, text_height) = cv2.getTextSize(txt, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = bbox_int[0]
        text_offset_y = bbox_int[1] - 2

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr_fill, cv2.FILLED)
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_brg_line, 1)
        cv2.putText(img, txt, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)
        
        ## uncertainty value display
        # get the width and height of the text box
        txt = uncertainty_top_text
        (text_width, text_height) = cv2.getTextSize(txt, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = bbox_int[0] + bbox_width - (text_width//2)
        text_offset_y = bbox_int[1] + text_height + 2

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr_fill, cv2.FILLED)
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_brg_line, 1)
        cv2.putText(img, txt, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)

        # get the width and height of the text box
        txt = uncertainty_left_text
        (text_width, text_height) = cv2.getTextSize(txt, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = bbox_int[0] 
        text_offset_y = bbox_int[1] + bbox_height + (text_height//2)

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr_fill, cv2.FILLED)
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_brg_line, 1)
        cv2.putText(img, txt, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)

        # get the width and height of the text box
        txt = uncertainty_right_text
        (text_width, text_height) = cv2.getTextSize(txt, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = bbox_int[2] - text_width - 2
        text_offset_y = bbox_int[1] + bbox_height + (text_height//2)

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr_fill, cv2.FILLED)
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_brg_line, 1)
        cv2.putText(img, txt, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)
        
        # get the width and height of the text box
        txt = uncertainty_bottom_text
        (text_width, text_height) = cv2.getTextSize(txt, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = bbox_int[0] + bbox_width - (text_width//2)
        text_offset_y = bbox_int[3] + (text_height//2) - 5

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr_fill, cv2.FILLED)
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_brg_line, 1)
        cv2.putText(img, txt, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)


    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img