# Introductory Course to PyTorch and TensorFlow


<p align="justify">
  Many data scientists and machine learning engineers frequently alternate between two main toolboxes: PyTorch and TensorFlow.  Both in Python, These  frameworks share many similarities, but also diverge in meaningful ways. These differences can make switching between the two frameworks cumbersome and inefficient. In this introductory course we try to learn Pytorch and Tensorflow in one go, In this way we not only hit two bird with one stone but also also understand the differences upfront, so we don't confuse only later on.
</p>

## Fundamentals
### Tensors
Both TensorFlow and PyTorch represent tensors as n-dimensional arrays of base datatypes, in both tensors as immutable objects.

#### Constant Tensors


*TensorFlow*

```python
tensor_tf = tf.constant([[1, 2],[3, 4],[5, 6]], dtype=tf.float16)
```


*PyTorch*  
In PyTorch you can create tensors on the desired device using the device attribute  
```python
tensor_torch = torch.HalfTensor([[1, 2],[3, 4],[5, 6]],device=torch.device('cpu'))
```
Tensors of different datatypes of  can be created in [PyTorch](https://pytorch.org/docs/stable/tensors.html) and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)

#### Random Tensors


*TensorFlow*

```python
tensor_tf = tf.random.uniform([4,4], minval=0, maxval=1, dtype=tf.float32, seed=1)
```


In PyTorch you can create tensors on the desired device using the device attribute

*PyTorch*
```python
tensor_torch = torch.rand(4,4,device=torch.device('cpu'),dtype=torch.float32)
```
#### Assignment to an element by index
In TensorFlow There is no straighforward way to assign to an element of a tensor by index, PyTorch allows it simply by 
```python
tensor_torch[1,1] = 2
```

### Variables and Parameters
Variables in TensorFlow and Parameters in PyTorch are gradient enabled tensors that combine along with operations to create the dynamic computational graph. Variables and Parameters are both mutable . 

*TensorFlow*
```python
variable_tf = tf.Variable(tensor_tf)
variable_tf[1,1].assign(2)
```

*PyTorch*
```python
parameter_torch = torch.nn.Parameter(tensor_torch)
with torch.no_grad():
  parameter_torch[1,1]=1
  ```
