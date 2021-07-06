# Introductory Course to PyTorch and TensorFlow


<p align="justify">
  Many data scientists and machine learning engineers frequently alternate between two main toolboxes: PyTorch and TensorFlow.  Both in Python, These  frameworks share many similarities, but also diverge in meaningful ways. These differences can make switching between the two frameworks cumbersome and inefficient. In this introductory course we try to learn Pytorch and Tensorflow in one go, In this way we not only hit two bird with one stone but also also understand the differences upfront, so we don't confuse only later on.
</p>

## Fundamentals
### Tensors

#### Constant Tensors


*TensorFlow*

```python
tf_tensor = tf.constant([[1, 2],[3, 4],[5, 6]], dtype=tf.float16)
```


*PyTorch*  
```python
torch_tensor = torch.HalfTensor([[1, 2],[3, 4],[5, 6]],device=torch.device('cpu'))
```
Tensors of different datatypes of  can be created in [PyTorch](https://pytorch.org/docs/stable/tensors.html) and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)

#### Random Tensors


*TensorFlow*

```python
tf_tensor = tf.random.uniform([4,4], minval=0, maxval=1, dtype=tf.float32, seed=1)
```
*PyTorch*
```python
torch_tensor = torch.rand(4,4,device=torch.device('cpu'),dtype=torch.float32)
```
#### Assignment to an element by index
In TensorFlow There is no straighforward way to assign to an element of a tensor by index while PyTorch allows it
```python
torch_tensor[1,1] = 2
```

### Variables and Parameters

*TensorFlow*
```python
tf_variable = tf.Variable(tf_tensor)
tf_variable[1,1].assign(2)
```

*PyTorch*
```python
torch_parameter = torch.nn.Parameter(torch_tensor)
with torch.no_grad():
  torch_parameter[1,1]=1
  ```

### Defining a new model


*TensorFlow*
```python
class LinearRegressionKeras(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.w = tf.Variable(tf.random.uniform(shape=[1], -0.1, 0.1))
    self.b = tf.Variable(tf.random.uniform(shape=[1], -0.1, 0.1))
    
  def __call__(self,x): 
    return x * self.w + self.b
```

*PyTorch*
```python
class LinearRegressionPyTorch(torch.nn.Module): 
  def __init__(self): 
    super().__init__() 
    self.w = torch.nn.Parameter(torch.Tensor(1, 1).uniform_(-0.1, 0.1))
    self.b = torch.nn.Parameter(torch.Tensor(1).uniform_(-0.1, 0.1))
  
  def forward(self, x):  
    return x @ self.w + self.b
  ```

