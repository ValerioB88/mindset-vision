# ImageNet Out-Of-Distribution classification

This methodology simply consists of testing whether a set of objects is correctly classified according to ImageNet classes. The annotation file of the selected dataset needs to contain a numerical column specifying what ImageNet class does the image belong to. For many of our datasets in the category `shape_and_object_recognition` this is column is automatically added as the `ImageNetClassIdx`. 

As usual, the [toml file](./default_classification_config.toml) contains all default optional arguments, with an explanation of what they do. The [example](./example/linedrawings_eval.py) will generate the linedrawing datasets and test a pre-trained `ResNet152` on them. 