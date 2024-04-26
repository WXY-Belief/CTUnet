# CTUnet: A Novel Paradigm Integrating CNNs and Transformers for Medical Image Segmentation
![示例图片](CTUnet.png)
We propose CTUnet, a novel network architecture that integrates Convolutional Neural Networks (CNNs) with Transformers. CTUnet primarily consists of two new modules, namely CTBlock and CTFusion. CTBlock integrates CNNs and Transformers in parallel to capture both semantic information and local details of images, thereby achieving a comprehensive representation of images. Additionally, CTFusion replaces the original skip-connection structure in Unet, utilizing spatial and channel attention mechanisms to merge and filter semantic information as well as local details. This process not only transfers local details to the decoder but also minimizes the semantic gap between the encoder and decoder.

# Training and testing
If you wish to utilize this network, you can employ this training framework [EasySegmentation]([https://openai.com/](https://github.com/WXY-Belief/EasySegmentation)) for training and testing.

# Citations
If you find this repo useful, please cite:
```
@misc{CTUnet,
      title={CTUnet: A Novel Paradigm Integrating CNNs and Transformers for Medical Image Segmentatio}, 
      author={Xiaoyu Wang, ChunLin Zhu, Jiaquan Li},
      year={2024},
      eprint={},
      archivePrefix={IJCNN2024},
      primaryClass={}
}
```
