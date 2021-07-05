# Introductory Course to PyTorch and TensorFlow


<p align="justify">
  Many data scientists and machine learning engineers frequently alternate between two main toolboxes: PyTorch and TensorFlow.  Both in Python, These  frameworks share many similarities, but also diverge in meaningful ways. These differences can make switching between the two frameworks cumbersome and inefficient. In this introductory course we try to learn Pytorch and Tensorflow in one go, In this way we not only hit two bird with one stone but also also understand the differences upfront, so we don't confuse only later on.
</p>

## Fundamentals
### Tensors
Both TensorFlow and PyTorch represent tensors as n-dimensional arrays of base datatypes

#### Constant Tensors


*TensorFlow*

```python
tf.constant([[1, 2],[3, 4],[5, 6]], dtype=tf.float16)
```


*PyTorch*  
In PyTorch you can create tensors on the desired device using the device attribute  
```python
torch.HalfTensor([[1, 2],[3, 4],[5, 6]],device=torch.device('cpu'))
```
Tensors of different datatypes of  can be created in [PyTorch](https://pytorch.org/docs/stable/tensors.html) and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)
#### Random Tensors


*TensorFlow*

```python
tf.random.uniform([4,4], minval=0, maxval=1, dtype=tf.float32, seed=1)
```


In PyTorch you can create tensors on the desired device using the device attribute

*PyTorch*
```python
torch.rand(4,4,device=torch.device('cpu'),dtype=torch.float32)
```
