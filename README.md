# ShapeNet Ensemble for 6D pose estimation:
My submission to the pose estimation track of OOD-CV at ECCV2022
### David Pitt

I train an ensemble of ResNet50-backbone CNN classifiers to predict each axis of rotation on each category. They are first trained on ImageNet1K with Richard Geirhos' style transfer augmentations, and then fine-tuned on the ROBIN dataset. 

I would like to note for the record that the [NeMo model](https://arxiv.org/pdf/2101.12378.pdf) is an inappropriate choice of baseline model for this specific challenge, as in order to train the model, you need CAD files from the PASCAL3D+ dataset that are not included in the ROBIN challenge repository. As such, I believe that the T-SWIN and SpecifiedResNet architectures make better baseline candidates for this track.

