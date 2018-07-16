# Linear Regression using Gradient Descent

The coding challenge is to implement gradient descent to find the line of best fit that predicts the relationship between 2 variables of your choice from a [kaggle](https://www.kaggle.com/datasets) dataset using python.

## Dependencies

<br /> numpy
<br /> pandas
<br /> matplotlib (vizualization)

Theoretically, **Gradient Descent** is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function. This iterative minimization is achieved using calculus, taking steps in the negative direction of the function gradient.

### Data
Dataset [Weather in Szeged 2006-2016](https://www.kaggle.com/budincsevity/szeged-weather) has been chosen for the challenge.

'weatherHistory.csv' has been chosen for the variables:
* Temperature
* Apparent Temperature

**Note :** This notebook contains the code for linear regression using Gradient Descent along with vizualizations.

### Data Vizualization

![Scatter](https://github.com/devrathmohanty/Coding_Challenge_Siraj/blob/master/Gradient_Descent/images/scatter.png)

Algorithm for **Sum of Squared Error**
```
def sumSquaredError(m, b, data):
    
    data = data.as_matrix()
    error = 0
    for i in range(data.shape[0]):
        
        temp = data[i, 1]  #column temperature
        appTemp = data[i, 0]  #column apparent temperature
        
        error += (appTemp - (m * temp + b)) ** 2
        
    sse = error/data.shape[0]
    return (sse)
```

### Vizualization after Linear Regression.
![Best Fit](https://github.com/devrathmohanty/Coding_Challenge_Siraj/blob/master/Gradient_Descent/images/bestFit.png)

Algorithm for **Gradient Descent**

```
def gradientDescentSteps(mStarting, bStarting, data, steps):
    
    m = mStarting
    b = bStarting
    
    displayFreq = steps//10
    
    for i in range(steps):
        m, b = stepGradient(m, b, data)
        if (i % displayFreq == 0):
            sse = sumSquaredError(m, b, data)
    
    return (m, b)
```

### Vizualization after Gradient Descent.
![Best Fit Gradient Descent](https://github.com/devrathmohanty/Coding_Challenge_Siraj/blob/master/Gradient_Descent/images/bestFitGradientDescent.png)

## References
[Siraj Raval Math of Intelligence](https://github.com/llSourcell/Intro_to_the_Math_of_intelligence)
<br /> [Introduction to Gradient Descent and Linear Regression - Matt Nedrich](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)
