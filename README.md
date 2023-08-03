## dugMatting: Decomposed-Uncertainty-Guided Matting [ICML 2023]

![Image](https://github.com/Fire-friend/dugMatting/blob/main/temp_fig/method.png?raw=true)

> ### News:
>
> [2023-8-3]: Publish the more detailed documentation.
>
> [2023-6-1]: Publish the rough code for studying dugMatting (MODNet) only.

## Abstract

Cutting out an object and estimating its opacity mask, known as image matting, is a key task in image and video editing. Due to the highly ill-posed issue, additional inputs, typically user-defined trimaps or scribbles, are usually needed to reduce the uncertainty. Although effective, it is either time consuming or only suitable for experienced users who know where to place the strokes. In this work, we propose a decomposed-uncertainty-guided matting (dugMatting) algorithm, which explores the explicitly decomposed uncertainties to efficiently and effectively improve the results. Basing on the characteristic of these uncertainties, the epistemic uncertainty is reduced in the process of guiding interaction (which introduces prior knowledge), while the aleatoric uncertainty is reduced in modeling data distribution (which introduces statistics for both data and possible noise). The proposed matting framework relieves the requirement for users to determine the interaction areas by using simple and efficient labeling. Extensively quantitative and qualitative results validate that the proposed method significantly improves the original matting algorithms in terms of both efficiency and efficacy.

## Usage

We built a standard framework that supports distributed training and easy to extend custom methods. The custom instruction can be seen in ***Custom instruction***.

**Training**:

1. Setup environment by `pip install -r requirement`.
2. Download the [P3M-10K](https://drive.google.com/uc?export=download&id=1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1) (Baidu Netdisk: [Link](https://pan.baidu.com/share/init?surl=X9OdopT41lK0pKWyj0qSEA)(pw: fgmc), [Agreement](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf) (MIT) ), and unzip to the directory of '*/data*'.
3. Modify the *config/ITMODNet_config.yaml* according to your experience (**Optional**). The *ITMODNet_config.yaml* contains the detailed comments for each parameter.
4. Run by `python public_worker --model ITMODNet`

The *checkpoint, log.txt, tensorboard file* can be seen in '/checkSave/'.

**Evaluation:**

`python evaluation `

## Custom instruction

The components of ***dataset, model, traner, evaluater, config***, etc. are separated. You can add your custom implementation refer to the releasing code (**strongly suggestion!!!**). There are some main explanations are as follows:

#### dataset (e.g., P3MP_dataset.py)

* The name of dataset file should follow **[custom]_dataset.py**.
* Put the **[custom]_dataset.py** into the **datasets** directory.
* The class name should follow **[custom]_Dataset**.
* The class should Include functions of ***get_train_data, get_val_data, get_show_data***.

#### Trainer (e.g., ITMODNet_trainer.py)

- Put the trainer file in the **traners** directory.
- The name of traner file should follow **[custom]_traner.py**.
- The function name should follow **[custom]_Trainer**.
- The returns of **[custom]_Trainer** should a loss dict, which requires the first key is *loss*.

## Statement

If you are interested in our work, please consider citing the following:

```
@inproceedings{,
author = {},
title = {dugMatting: Decomposed-Uncertainty-Guided Matting},
year = {},
isbn = {},
publisher = {},
address = {},
url = {},
doi = {},
booktitle = {},
pages = {},
numpages = {},
keywords = {},
location = {},
series = {}
}
```

This project is under MIT licence.
