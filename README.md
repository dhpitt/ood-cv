# ShapeNet Ensemble for 6D pose estimation:
My submission to the pose estimation track of OOD-CV at ECCV2022

---

I train an ensemble of ResNet50-backbone CNN classifiers to predict each axis of rotation on each category. They are first trained on ImageNet1K with Richard Geirhos' style transfer augmentations (strong augmentations which are explicitly allowed as per the contest guidelines), and then fine-tuned on the ROBIN dataset. I use a loss function that is a combination of classification loss (cross-entropy) and a contrastive term that pushes representations of images as close as possible to the same image cropped to its bounding box.

---

## Model training and inference

To train the submitted bounding-box contrastive learning model on ROBIN, first unzip the full ROBIN dataset to some path ROBIN_PATH. Then run the following:

```
$ cd second
$ python main_pl_bbox.py --train_root PATH_TO_ROBIN/train/ --val_root PATH_TO_ROBIN/iid_test/ --anno_root PATH_TO_ROBIN/annotations/
```

Since every class and every aspect of pose (azimuth, elevation, inplane rotation and distance) gets its own model, this takes 18-24 hours to run on an NVIDIA RTX 3090. Checkpoints will save to `./checkpoints_bbox/{class}_{pose aspect}`. 

To test this model on unlabeled data, run the following script. It is written to handle the phase 2 pose data, so it expects images under a folder labeled `/images/` with names like `string_of_numbers_{class_name}_more_numbers.jpg`:

```
$ python inference_pl_p2.py --test_root PATH_TO_UNLABELED_DIR/images/ --label_dest SOME_DIR --ckpt_dir PATH_TO_SAVED_MODELS
```

This will load models from ./PATH_TO_SAVED_MODELS and save predictions to a file called pred.csv under a ./LABEL_DIR specified in the command.

