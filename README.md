# Unofficial implementation of Design Booster 
This is unofficial implementation of [Design Booster](https://arxiv.org/abs/2302.02284) by [mkshing](https://twitter.com/mk1stats).

![paper](paper.png)

> Diffusion models are able to generate photorealistic images in arbitrary scenes. However, when applying diffusion models to image translation, there exists a trade-off between maintaining spatial structure and high-quality content. Besides, existing methods are mainly based on test-time optimization or fine-tuning model for each input image, which are extremely time-consuming for practical applications. To address these issues, we propose a new approach for flexible image translation by learning a layout-aware image condition together with a text condition. Specifically, our method co-encodes images and text into a new domain during the training phase. In the inference stage, we can choose images/text or both as the conditions for each time step, which gives users more flexible control over layout and content. Experimental comparisons of our method with state-of-the-art methods demonstrate our model performs best in both style image translation and semantic image translation and took the shortest time.

