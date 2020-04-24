# CS7650-project-vqa

### Data: 
1. Dataset consists of ResNet-18 Image features, Tokenised Questions, Tokenised Answers (in Numpy format): https://drive.google.com/open?id=1DCHNVK5pxAKOiiJcfMGcO-pSjJDIv1zT
2. Dataset containing Image locations, Question and Answers can be found in the dataset directory. 
3. For generating VGG-16 Image features needed by Parallel and Alternate Co-Attention, Fusion and Co-Attention model use the Python notebook named DataCreationVGG.
4. Raw Images for dataset (needed for creating VGG-16 Image features): http://images.cocodataset.org/zips/train2014.zip, http://images.cocodataset.org/zips/val2014.zip

### Models:
1. Fusion Model: Run Trainer.ipynb with default settings
2. CNN + LSTM Model: Run Trainer.ipynb and change fusion_type to Concatenation
3. Parallel and Alternate Co-Attention Model: Run ParallelAndAlternateCoAttention.py
4. Fusion and Co-Attention Model: Run FusionAndCoAttention.py

### Model Weights: 
Trained Weights for all models: https://drive.google.com/open?id=1dMvJ9bQlIhEHjr_zw5uL6KLuGJfKpfqs
