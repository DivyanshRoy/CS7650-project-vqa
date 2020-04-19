# CS7650-project-vqa

#### Dataset with 1000 examples per class: https://drive.google.com/open?id=1mlTiL_MMOYM1ohecJPjKpISr13b16Oup

#### Updated Dataset: https://drive.google.com/open?id=1SNIdZWokJkPCWDAwI6Rir44hiCNDQzYY
VGG Feature Map: https://drive.google.com/open?id=17VVNSCvntdrkIFEINB9CThhtpFLAVtQs

#### Images (~745 MB): Training (5k), Validation (1k)
https://drive.google.com/file/d/1cLUCEGM4UW_GrI5iZdZfdLaPjM0hYvx_/view

Get combined_train_input.npy from https://drive.google.com/open?id=1_PkcsLndQb42SvwXRTU6SK-9_TvoKfVh and paste it in the preprocessed_data folder

Get train_image_list.npy from https://drive.google.com/open?id=12zyK8Joa8F0CtpwVWW4alcY7cK-q22h9

Get val_image_list.npy from https://drive.google.com/open?id=16L8NwE7S48Q2LyPtWqwU8L0Tv8KZCdRw

Get resnet_152_train_image_features.npy from https://drive.google.com/open?id=1XgLY5lZZ08p2J_G5juRzBEs-T83HjKWp

Get resnet_152_val_image_features.npy from https://drive.google.com/open?id=1sZEZ2stpsQu470emis1DMHftGXCL5E5E

<br><br>
#### Pre-processed data:
###### train_dataset_5k.json: Contains 5k randomly chosen question, answer, image triplets from the training dataset.
###### val_dataset_1k.json: Contains 1k randomly chosen question, answer, image triplets from the validation dataset.

###### answer_tokens_top1k.json: Contains the Top 1k answers (sorted by frequency) from the training dataset in tokenised form.
###### train_dataset_5k_tokenised.json: Contains most common answer (amongst 10 answers for each question with count >= 3) in tokenised form using tokenisation dictionary from answer_tokens_top1k.json
###### val_dataset_1k_tokenised.json: Contains most common answer (amongst 10 answers for each question with count >= 3) in tokenised form using tokenisation dictionary from answer_tokens_top1k.json
