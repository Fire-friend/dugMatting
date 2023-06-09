## dugMatting: Decomposed-Uncertainty-Guided Matting [ICML 2023]

![Image](https://github.com/Fire-friend/dugMatting/blob/main/temp_fig/method.png?raw=true)

> ### News:
>
> [2023-6-1]: Publish the rough code for studying dugMatting (MODNet) only.

## Abstract

Cutting out an object and estimating its opacity mask, known as image matting, is a key task in image and video editing. Due to the highly ill-posed issue, additional inputs, typically user-defined trimaps or scribbles, are usually needed to reduce the uncertainty. Although effective, it is either time consuming or only suitable for experienced users who know where to place the strokes. In this work, we propose a decomposed-uncertainty-guided matting (dugMatting) algorithm, which explores the explicitly decomposed uncertainties to efficiently and effectively improve the results. Basing on the characteristic of these uncertainties, the epistemic uncertainty is reduced in the process of guiding interaction (which introduces prior knowledge), while the aleatoric uncertainty is reduced in modeling data distribution (which introduces statistics for both data and possible noise). The proposed matting framework relieves the requirement for users to determine the interaction areas by using simple and efficient labeling. Extensively quantitative and qualitative results validate that the proposed method significantly improves the original matting algorithms in terms of both efficiency and efficacy.

## Usage

**Training**:

We built a standard framework that supports distributed training and easy to extend custom methods. 

You can modify the  *config/ITMODNet_config.yaml* and then run as follows:

`python public_worker --model ITMODNet`

**Evaluation:**

`python evaluation `

------

## Custom guideline

The components of *dataloader, model, traner, evaluater, config*, etc. are separated. If you want to add a custom method or loader, the main guideline is as follows:
