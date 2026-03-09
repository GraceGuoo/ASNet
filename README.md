# ASNet
This project provides the code and results for 'ASNet: An adaptive scene-aware network for RGB-thermal urban scene semantic segmentation'. 

Thank you for your interest. The complete training scripts and details will be made publicly available upon publication.

## Usage
### Requirements
1. Python 3.7.15
2. PyTorch 1.11.0
3. CUDA 11.3+

and
    pip install -r requirements.txt

>Note: For a detailed environment configuration tutorial you can refer to our CSDN [blog](https://blog.csdn.net/qq_41973051/article/details/128844400?spm=1001.2014.3001.5501).

### Datasets

<details>
  <summary>dataset</summary>
  <ul>
    <li>mfnet/
        <ul>
        <li>RGB/
      <ul>
        <li>name1.png</li>
        <li>name2.png</li>
      </ul>
    </li>
    <li>TH/
      <ul>
        <li>……</li>
      </ul>
    </li>
    <li>Labels/
      <ul>
        <li>……</li>
      </ul>
    </li>
  </ul>
    </li>
    <li>PST900/
      <ul>
        <li>……</li>
      </ul>
    </li>
  </ul>
</details>

If you would like to download the dataset, please contact me.

### Pretrain weights:
Download the pretrained Resnet-152 [here](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet152-b121ed2d.pth).

### Config
The parameters of the dataset and the network can be modified through this config file.

## Results
### Results on MFNet (9-class):

| Backbone  | Modal | mAcc | mIoU   |
|--------|------|------|--------|
| Resnet-152  | RGB-T   | 77.8 | 55.3 |

### Results on PST900 (5-class):

| Backbone  | Modal | mAcc | mIoU   |
|--------|------|------|--------|
| Resnet-152  | RGB-T   | 82.85 | 77.08 |

All code will be made public after the paper is published.
