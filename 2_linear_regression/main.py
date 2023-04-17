'''
TensorFlow Core Learning Algorithms
In this notebook we will walk through 4 fundemental machine learning algorithms. We will apply each of these algorithms to unique problems and datasets before highlighting the use cases of each.

The algorithms we will focus on include:

Linear Regression
Classification
Clustering
Hidden Markov Models
It is worth noting that there are many tools within TensorFlow that could be used to solve the problems we will see below. I have chosen the tools that I belive give the most variety and are easiest to use.
'''



'''
Linear Regression
Linear regression is one of the most basic forms of machine learning and is used to predict numeric values.

In this tutorial we will use a linear model to predict the survival rate of passangers from the titanic dataset.

This section is based on the following documentation: https://www.tensorflow.org/tutorials/estimator/linear
'''



'''
Linear Regression
Linear regression is one of the most basic forms of machine learning and is used to predict numeric values.

In this tutorial we will use a linear model to predict the survival rate of passangers from the titanic dataset.

This section is based on the following documentation: https://www.tensorflow.org/tutorials/estimator/linear
'''


import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
# plt.show()

'''
We can see that this data has a linear coorespondence. When the x value increases, so does the y. Because of this relation we can create a line of best fit for this dataset. In this example our line will only use one input variable, as we are working with two dimensions. In larger datasets with more features our line will have more features and inputs.

"Line of best fit refers to a line through a scatter plot of data points that best expresses the relationship between those points." (https://www.investopedia.com/terms/l/line-of-best-fit.asp)

Here's a refresher on the equation of a line in 2D.

y=mx+b 

Here's an example of a line of best fit for this graph.
'''



plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), y)
plt.show()

'''
Here's a breakdown of the code:

    plt.plot(): This is a Matplotlib function that creates a line plot of the data.

    np.unique(x): This function returns the sorted unique elements of the array x. This is useful because it removes any duplicates and sorts the array in ascending order.

    np.polyfit(x, y, 1): This function fits a polynomial of degree 1 (i.e., a straight line) to the data points represented by the arrays x and y. The resulting polynomial coefficients are returned as a NumPy array.

    np.poly1d(np.polyfit(x, y, 1)): This function converts the polynomial coefficients into a callable polynomial function that can be evaluated at any point.

    np.poly1d(np.polyfit(x, y, 1))(np.unique(x)): This code evaluates the polynomial function at the sorted unique elements of x, which gives us the predicted y-values for each corresponding x-value.

    The resulting x and y values are then passed as arguments to plt.plot(), which creates a line plot of the data points and the linear regression line.

    So, in summary, this line of code creates a scatter plot of the data points represented by the arrays x and y, and adds a linear regression line to the plot using the np.polyfit() and np.poly1d() functions.
    
    
    
    
    
    
    
    
First, let's look at the np.unique(x) part of the code. This function returns the unique elements of the array x, in sorted order. The reason for using np.unique() is to ensure that we only use each value of x once, and to ensure that the values are sorted in ascending order. This will be useful later on when we plot the linear regression line.


Next, let's look at the np.polyfit(x, y, 1) part of the code. This function performs a linear regression on the data points represented by the arrays x and y, and returns the coefficients of the resulting polynomial. The 1 argument tells np.polyfit() that we want to fit a straight line (i.e., a polynomial of degree 1) to the data. The resulting coefficients are returned as a NumPy array of length 2, representing the slope and y-intercept of the line.


The coefficients returned by np.polyfit() can be used to create a polynomial function that represents the linear regression line. This is done using the np.poly1d() function, which takes the coefficients as its argument and returns a callable polynomial function. The resulting function can be evaluated at any value of x to obtain 
the corresponding predicted value of y.


Finally, the plt.plot() function is used to plot the linear regression line. The x values passed to plt.plot() are the unique values of x returned by np.unique(). The y values passed to plt.plot() are the predicted values of y obtained by evaluating the linear regression function at the corresponding x values. These predicted values are obtained by calling the polynomial function returned by np.poly1d() with the unique x values as its argument.


Overall, this line of code generates a scatter plot of the data points represented by the arrays x and y, and adds a linear regression line to the plot using the np.polyfit() and np.poly1d() functions. The resulting plot shows the relationship between the x and y variables, and provides a visual representation of the linear regression analysis.
'''
