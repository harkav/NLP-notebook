# Euclidean distance 

"""
# Euclidean Distance

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

## **Lengdenormalisering**
$$\frac{x}{||x||}$$

**Length of a vector**
$$||x|| = \sqrt{x \cdot x} = \sqrt{\sum_{i=1}^nx_i^2}$$