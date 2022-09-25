# ShapeNet Ensemble for 6D pose estimation:
My submission to the pose estimation track of OOD-CV at ECCV2022

---

I train an ensemble of ResNet50-backbone CNN classifiers to predict each axis of rotation on each category. They are first trained on ImageNet1K with Richard Geirhos' style transfer augmentations (strong augmentations which are explicitly allowed as per the contest guidelines), and then fine-tuned on the ROBIN dataset. 

---

## Model training and inference

To train the submitted model on ROBIN, first unzip the full ROBIN dataset to some path ROBIN_PATH. Then run the following:

```
$ cd baseline
$ python main_pl.py --train_root PATH_TO_ROBIN/train/ --val_root PATH_TO_ROBIN/iid_test/
```

Since every class and every aspect of pose (azimuth, elevation, inplane rotation and distance) gets its own model, this takes 12-16 hours to run on an NVIDIA RTX 3090. Checkpoints will save to `./checkpoints/{class}_{pose aspect}`. 

To test this model on unlabeled data, run the following script. It is written to handle the phase 2 pose data, so it expects images under a folder labeled `/images/` with names like `string_of_numbers_{class_name}_more_numbers.jpg`:

```
$ python inference_pl_p2.py --test_root PATH_TO_UNLABELED_DIR/images/
```

This will save a file called pred.csv to a directory specified in the script.

