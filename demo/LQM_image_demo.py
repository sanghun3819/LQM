from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIAlign, RoIPool
import warnings
import torch

import numpy as np
import cv2

from mmcv.visualization.image import *

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file', default = 'configs/uncertainty_guide/uncertainty_guide_r50_fpn_1x.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default ='work_dirs/uncertainty_test/latest.pth')
    # parser.add_argument('--img', help='Image file', default = 'demo/demo.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000578978.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000569118.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000579228.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000581427.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000581494.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000578739.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000000178.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000007537.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000581269.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000574777.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000171457.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000189629.jpg')

    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000244947.jpg')
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000247852.jpg') #bad detection example
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000264192.jpg') # surf
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000264192.jpg') # surf2

    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000579128.jpg') # surf - test1.jpg
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000578754.jpg') # test2
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000580239.jpg') # test3
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000472144.jpg') # test4 - giraffe
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000122278.jpg') # test5 humans
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000242267.jpg') # test6
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000295089.jpg') # test7
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000386980.jpg') # test8 tennis
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000285286.jpg') # test9 tennis
    # parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000391692.jpg') # test10 traffic sign
    parser.add_argument('--img', help='Image file', default = 'data/coco/test2017/000000011245.jpg') # test11
    # parser.add_argument('--img', help='Image file', default = 'data/coco/val2017/000000052007.jpg') # test12
    # parser.add_argument('--img', help='Image file', default = 'data/coco/val2017/000000555972.jpg') # test13
    # parser.add_argument('--img', help='Image file', default = 'data/coco/val2017/000000245448.jpg') # test14
    # parser.add_argument('--img', help='Image file', default = 'data/coco/val2017/000000315450.jpg') # test15
    # parser.add_argument('--img', help='Image file', default = 'data/coco/val2017/000000372203.jpg') # test16
    # parser.add_argument('--img', help='Image file', default = 'data/coco/val2017/000000555009.jpg') # test17
    


    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a single image
    # result = inference_detector(model, args.img)
    result, uncertainty, quality = _inference_detector(model, args.img)


    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    model.show_result(args.img, result,
                            score_thr=args.score_thr,
                            bbox_color = 'green',
                            text_color = (0, 255, 255),
                            thickness = 2,
                            font_scale = 1.0,
                            win_name='',
                            show=True,
                            wait_time=0,
                            out_file='demo/detection_result.jpg')
    
    img = cv2.imread(args.img)
    feature_stride = 3 # 0 ~ 4 stride index # 1 : 50,76 --> -9 right

    #quality visualize
    # N, C, H, W = quality[feature_stride].shape
    # feat = quality[feature_stride].reshape(-1, H, W)
    # feature_map = cv2.resize(feat[0].cpu().numpy(), (img.shape[1], img.shape[0]))
    # featureshow = None
    # featureshow = cv2.normalize(feature_map, featureshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # feature_map = cv2.applyColorMap(featureshow, cv2.COLORMAP_JET)
    # cv2.imwrite('demo/quality.jpg', feature_map)
    # overlay = cv2.addWeighted(img, 0.4, feature_map, 0.3, 0)
    # cv2.imwrite('demo/overlay/quality_overlay.jpg', overlay)

    feature_map = []
    for stride, _ in enumerate(quality) :
        N, C, H, W = quality[stride].shape
        feature = cv2.resize(quality[stride].reshape(H, W).cpu().numpy(), (img.shape[1], img.shape[0]))
        feature_map.append(feature)
    
    feature_map = feature_map[0] + feature_map[1] + feature_map[2] + feature_map[3] + feature_map[4]  
    featureshow = None
    featureshow = cv2.normalize(feature_map, featureshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    feature_map = cv2.applyColorMap(featureshow, cv2.COLORMAP_JET)
    cv2.imwrite('demo/potential_quality.jpg', feature_map)
    overlay = cv2.addWeighted(img, 0.4, feature_map, 0.3, 0)
    cv2.imwrite('demo/potential_quality_overlay.jpg', overlay)

    # uncertainty feature visualize
    N, C, H, W = uncertainty[feature_stride].shape
    feat = (1.0 - uncertainty[feature_stride]).reshape(-1, H, W)

    mask = torch.ones_like(feat).cuda()
    mask_ratio = 0.15
    mask_ratio_opposite = 0.1
    feat_ratio = 0.15

    for i in range(4):
        mask[i][:, :round(W*mask_ratio)] = 0.  #left
        mask[i][:round(H*mask_ratio), :] = 0. 
        mask[i][:, -round(W*mask_ratio):] = 0. #right
        mask[i][-round(H*mask_ratio):, :] = 0. #bottom
        #mask[i][:, :round(W*0.4)] = 0.  #left
        #mask[i][:round(H*0.3), :] = 0.  #top
        #mask[i][:, -round(W*0.5):] = 0. #right
        #mask[i][-round(H*0.22):, :] = 0. #bottom


    # mask[1][:round(H*mask_ratio), :] = 0.  #top
    # mask[1][-round(H*mask_ratio_opposite):, :] = 0.

    # mask[2][:, -round(W*mask_ratio):] = 0. #right
    # mask[2][:, :round(W*mask_ratio_opposite)] = 0.

    # mask[3][-round(H*mask_ratio):, :] = 0. #bottom
    # mask[3][:round(H*mask_ratio_opposite), :] = 0.

    # feat = torch.where(feat>0.6, feat, torch.tensor(0.).cuda())
    feat *= mask
    feat = torch.where(feat>0.6, feat, feat*feat_ratio)
    # feat = torch.where(feat>0.6, feat*0, feat*0)

    #feat *= mask
    # feat[2] = feat[2]*mask_r

    # feature_map = cv2.resize(feat[0].cpu().numpy(), (img.shape[1], img.shape[0]))
    # featureshow = None
    # featureshow = cv2.normalize(feature_map, featureshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # feature_map = cv2.applyColorMap(featureshow, cv2.COLORMAP_JET)
    # cv2.imwrite('demo/left_uncertainty.jpg', feature_map)
    # overlay = cv2.addWeighted(img, 0.4, feature_map, 0.3, 0)
    # cv2.imwrite('demo/overlay/left_overlay.jpg', overlay)

    # feature_map = cv2.resize(feat[1].cpu().numpy(), (img.shape[1], img.shape[0]))
    # featureshow = None
    # featureshow = cv2.normalize(feature_map, featureshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # feature_map = cv2.applyColorMap(featureshow, cv2.COLORMAP_JET)
    # cv2.imwrite('demo/top_uncertainty.jpg', feature_map)
    # overlay = cv2.addWeighted(img, 0.4, feature_map, 0.3, 0)
    # cv2.imwrite('demo/overlay/top_overlay.jpg', overlay)

    # feature_map = cv2.resize(feat[2].cpu().numpy(), (img.shape[1], img.shape[0]))
    # featureshow = None
    # featureshow = cv2.normalize(feature_map, featureshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # feature_map = cv2.applyColorMap(featureshow, cv2.COLORMAP_JET)
    # cv2.imwrite('demo/right_uncertainty.jpg', feature_map)
    # overlay = cv2.addWeighted(img, 0.4, feature_map, 0.3, 0)
    # cv2.imwrite('demo/overlay/right_overlay.jpg', overlay)

    # feature_map = cv2.resize(feat[3].cpu().numpy(), (img.shape[1], img.shape[0]))
    # featureshow = None
    # featureshow = cv2.normalize(feature_map, featureshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # feature_map = cv2.applyColorMap(featureshow, cv2.COLORMAP_JET)
    # cv2.imwrite('demo/bottom_uncertainty.jpg', feature_map)
    # overlay = cv2.addWeighted(img, 0.4, feature_map, 0.3, 0)
    # cv2.imwrite('demo/overlay/bottom_overlay.jpg', overlay)
   
    

def _inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    # prepare data
    data = dict(img_info=dict(filename=img), img_prefix=None)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        return model(return_loss=False, rescale=True, demo=True, **data)
    
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    
    model.show_result(args.img, result,
                            score_thr=args.score_thr,
                            bbox_color = 'green',
                            text_color = 'red',
                            thickness = 1,
                            font_scale = 0.4,
                            win_name='',
                            show=False,
                            wait_time=0,
                            out_file='demo/test.jpg')

if __name__ == '__main__':
    main()
