# Dot product 

$a \cdot b =\sum_{i=0}^{n - 1}(a_ib_i)$


In other words, for each feature in vector a and b, take the sum of the product of feature i in vector a with feature i in vector b from index 0 to the last index of the vectors. (Where we count index 0 as the first index of a vector.) The algorithm assumes that the vectors are of equal length. 

Example: 

a = [1, 2, 3]

b = [2, 2, 2]

dot product = ((1 * 2) + (2 * 2) + (3 * 2) ) = 12

Example 2: 

a = [1, 1, 7]

b = [2, 3, 6]

dot product = ((1 * 2) + (1 * 3) + (7 * 6) ) = 50


Example 3: 

a = [0]

b = [1]

dot procut = 0 * 1 = 0


Here is an implementation in Python, meant to be readable. 

```Python

    def dot_product(vector1: list[float], vector2 : list[float]) -> float :
    """
    A method for finding the dot product of two vectors.
    
    Args: 
        vector1 (list[float]): a list representing a vector.
        vector2 (list[float]): a list representing a different vector. 
        
    Returns: 
        sum (float): the sum of the calculation.
        
    Raises: 
        ValueError: If the vectors are not of the same length. 
    """
    
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
        
    
    total = 0 
    for v1, v2 in zip(vector1, vector2): 
        total += (v1 * v2)
    return total



```




# Euclidean distance 



The Euclidean distance between two vectors **a** and **b** is calculated as:

$$
d(a, b) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$


We can do this in a very straightforward way in Python like this: 

```python

    def euclidean_distance(
        vector1: list[float], vector2 : list[float]
        ) -> float :

        if len(vector1) != len(vector2):
            raise ValueError("Vectors must be of the same length")
            
        
        total = 0 
        for v1, v2 in zip(vector1, vector2): 
            total += (v1 - v2)**2
        return math.sqrt(total)

```


The body of the method can also be rewritten using a list comprehension in the following way 

```python

    def euclidean_distance(vector1, vector2): 
        return math.sqrt(sum( x - y) ** 2 for x, y in zip(vector1, vector2))
```

## **Length normalization**
$$\frac{x}{||x||}$$

**Length of a vector**
$$||x|| = \sqrt{x \cdot x} = \sqrt{\sum_{i=1}^nx_i^2}$$


```python

import math

def length_normalization(vector: list[float]) -> list[float] :
    """
    A method for normalizing a vector.
    
    Args: 
        vector1 (list[float]): a list representing a vector.
         
        
    Returns: 
        normalized_vector (list[float]): the sum of the calculation.
        
    Raises: 
        ValueError: If it is a zero-length vector. 
    """
    
 
    
    total = 0
    for element in vector: 
        total += element ** 2
    length = math.sqrt(total)
    
    if length == 0: 
        raise ValueError("cannot normalize a zero-length vector")
    normalized_vector = [x/ length for x in vector]

    return normalized_vector


```