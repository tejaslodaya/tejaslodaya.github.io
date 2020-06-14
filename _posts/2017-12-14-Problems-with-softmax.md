---
layout: post
title: Problems with softmax
---

*   [Introduction](#introduction)
	*   [Multinoulli Distribution](#multinoulli)
	*   [Softmax Regression](#softmax)
*   [Underflow and Overflow](#undernover)
*   [Conclusion](#conclusion)
*   [References](#reference)

<h3 id="introduction">Introduction</h3>
---

*	<h6 id="multinoulli"> Multinoulli Distribution </h6>
The multinoulli, or categorical, distribution is a distribution over a single discrete variable with `k` diﬀerent states, where `k` is ﬁnite. Multinoulli distributions are often used to refer to distributions over categories of objects.

*	<h6 id="softmax"> Softmax Regression </h6>
The softmax function is used to predict the probabilities associated with a multinoulli distribution. Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes. In logistic regression we assumed that the labels were binary: `y(i)∈{0,1}`. We used such a classifier to distinguish between two kinds of hand-written digits. Softmax regression allows us to handle `y(i)∈{1,…,K}` where K is the number of classes. In the special case where `K=2`, one can show that softmax regression reduces to logistic regression.

	![](https://render.githubusercontent.com/render/math?math=softmax(x)_i=%20%5Cfrac{exp(x_i)}{\%20%5Csum_{j=1}^n%20%5Cexp(x_j)}&mode=display)
	
<h3 id="undernover"> Underflow and Overflow</h3>
---

Underflow occurs when numbers near zero are rounded to zero. We usually want to avoid division by zero or taking the logarithm of zero.
Overflow occurs when numbers with large magnitude are approximated as ∞ or -∞. Further arithmetic will usually change these infinite values into not-a-number values. 

Using equation above, let's consider these scenarios:

1. When all ![](https://render.githubusercontent.com/render/math?math=x_i&mode=display) are equal to some constant `c`. Analytically, all outputs should be equal to `1/n`. Numerically, this may not occur when `c` has a large magnitude. If `c` is very negative, exp(c) will turn to zero (underflow). This means denominator of the softmax will be 0, making the final result undefined.
2. When `c` is very large and positive, exp(c) will turn to infinity (overflow), again resulting in the expression as a whole being undefined.

Both of these difficulties can be resolved by instead evaluating softmax(z) where ![](https://render.githubusercontent.com/render/math?math=z%20=%20x%20-%20max_i%20(x_i)&mode=display). Simple algebra shows that the value of softmax function does not change analytically by adding or subtracting a scalar from the input vector. 
Let's take an example where 


	x = [1, 2, 3, 4, 5]
	max(x) = 5
	z = x - max(x) = [-4, -3, -2, -1, 0]


Possibility of overflow is ruled out since **largest** argument to exp is 0 and `exp(0) = 1`. A possibility of underflow is also ruled out since at least one term in the denominator has a value of 1.

<h3 id="conclusion"> Conclusion </h3>
---

For the most part, developers of low-level libraries will keep in mind when implementing deep learning algorithms. In some cases, it is possible to implement a new algorithm and have the new implementation automatically stabilized. Theano, Tensorflow and Caffe are examples of software packages that automatically detect and stabilize many common numerically unstable expressions that arise in the context of deep learning.

<h3 id="reference"> References </h3>
---

1. [http://www.deeplearningbook.org/contents/numerical.html](http://www.deeplearningbook.org/contents/numerical.html)
2. [http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)