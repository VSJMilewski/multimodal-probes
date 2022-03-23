# Data Dir

Data folder follows the patern of the
[Data Science for Social Good](https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow) guide.

There are three subfolders:
 - raw: here we store all the files directly how it is downloaded (and extracted) from the `distributions/webpages`.
 - intermediate: while preprocessing, several formats can be constructed. If it is not a final format that is used, it is stored here.
 - processed: the final processed files. These are all the files that are needed to run the experiments.

You have to fill the raw folder with the needed datasets. By default, the codebase supports
**Penn Treebank 3** and **Flickr30K**.

The intermediate and processed folder are filled by the preprocessing script. If preprocessing is complete,
and you want to free space, you are free to clear the raw and the intermediate folder.

See the file structure at the bottom of this README to see what the default expected structure and data is.

# PTB3
the PTB3 dataset is a text-only dataset. You need to get access and download the raw dataset from the
University of Pennsylvania [LDC data catalog](https://catalog.ldc.upenn.edu/LDC99T42).

# FLICKR30K
the Flickr30K dataset is an image caption dataset witk 30k image and 5 captions each. You need to get access and download the raw images.
Fill in [this form](https://forms.illinois.edu/sec/229675) to request access to the Flickr30k images.  
For more details, see the [dataset web page](http://shannon.cs.illinois.edu/DenotationGraph/). Download and extract the zip into *raw/Flickr30k/*

We also need image regions linked to the sentence, and we need the dependency tree-structure for training and testing our probes.
 - For the regions, we use the [Flickr30k-Entities](http://bryanplummer.com/Flickr30kEntities/).
   Clone their [GitHub Page](https://github.com/BryanPlummer/flickr30k_entities) into *raw/Flickr30k/*.
 
## Collecting Image Features
To work with Flickr30K, we cannot simply rely on the single datamodule.
We require a docker image and a special script to extract the image features.
Here are the complete instructions for extracting image features for the Flickr30K data.

### Extract Image Features
From the `multimodal-probes` repository root, first use `airsplay/bottom-up-attention`
to extract image features with Faster R-CNN.
For example:
```text
# Path to training images
mkdir data/intermediate/Flickr30k
docker run \
    --gpus all \
    -v $(pwd)/scripts:/workspace/scripts \
    -v $(pwd)/data/raw/Flickr30k:/workspace/data:ro \
    -v $(pwd)/data/intermediate/Flickr30k:/workspace/features \
    -v $(pwd)/caffe_pretrained:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd scripts/
CUDA_VISIBLE_DEVICES=0 python extract_flickr30k_images.py
```

If you want to use spacy created dependency trees, change the commands to:
```text
# Path to training images
mkdir data/intermediate/Flickr30k/spacy
docker run \
    --gpus all \
    -v $(pwd)/scripts:/workspace/scripts \
    -v $(pwd)/data/raw/Flickr30k:/workspace/data:ro \
    -v $(pwd)/data/intermediate/Flickr30k/spacy:/workspace/features \
    -v $(pwd)/caffe_pretrained:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd scripts/
CUDA_VISIBLE_DEVICES=0 python extract_flickr30k_images.py --use_spacy
```

### Serialize Image Features & Extract Captions
Now you can simply continue running of the main script and the data_module.
