# Introductory Course to PyTorch and TensorFlow


<p align="justify">
  Many data scientists and machine learning engineers frequently alternate between two main toolboxes: PyTorch and TensorFlow.  Both in Python, These  frameworks share many similarities, but also diverge in meaningful ways. These differences can make switching between the two frameworks cumbersome and inefficient. In this introductory course we try to learn Pytorch and Tensorflow in one go, In this way we not only hit two bird with one stone but also also understand the differences upfront, so we don't confuse only later on.
</p>

## Fundamentals
### Tensors


Both TensorFlow and PyTorch represent tensors as n-dimensional arrays of base datatypes

> TensorFlow
```python
tf.constant([[1, 2],[3, 4],[5, 6]], dtype=tf.float16)
```

In PyTorch you can create tensors on the desired device using the device attribute
> PyTorch
```python
torch.ones([2, 4], dtype=torch.float64, device=torch.device('cpu'))
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RoozbehSanaei/PyTorch-TensorFlow/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
