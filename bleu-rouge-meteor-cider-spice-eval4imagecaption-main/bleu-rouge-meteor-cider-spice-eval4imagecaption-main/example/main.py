from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

annotation_file = r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/swin/references_swin9.json'
results_file = r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/Image-Captioning/Captions/swin/captions_swin9.json'
# annotation_file = 'state_dict/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/example/references_vgg.json'
# results_file = 'state_dict/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/example/captions_vgg.json'
# annotation_file = 'state_dict/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/example/references_vit.json'
# results_file = 'state_dict/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/example/captions_vit.json'
# annotation_file = 'state_dict/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/example/references.json'
# results_file = 'state_dict/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/bleu-rouge-meteor-cider-spice-eval4imagecaption-main/example/captions.json'

# create coco object and coco_result object
coco = COCO(annotation_file)
coco_result = coco.loadRes(results_file)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
#coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')
