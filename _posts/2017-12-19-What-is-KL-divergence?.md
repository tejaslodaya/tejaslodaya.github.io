---
layout: post
title: What is Kullback-Leibler divergence?
---

*   [Background](#background)
	*   [Entropy](#entropy)
	*   [Information Gain](#infogain)
	*   [Surprise Factor](#surprise)
*   [Cross Entropy](#xentropy)
*   [Kullback - Leiber divergence](#kldivergence)
*   [Facts about KL divergence](#facts)
*   [Usage in Machine Learning](#usage)
*   [References](#references)


<h3 id="background">Background</h3>
---

*	<h6 id="entropy"> Entropy </h6>
	Entropy is the expected amount of information when an event is drawn from a distribution. Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy. Shannon entropy is given by: 

	![](https://render.githubusercontent.com/render/math?math=H(x)%20=%20%5Cmathbb{E}_{x~P}[I(x)]%20=%20-%5Cmathbb{E}_{x~P}[log%20P(x)]%20=%20-%5Csum_{n=1}^{k}%20P(x)%20*%20log(P(x))&mode=display)

	The expectation in this case is probability-weighted average on all possible k states.
*	<h6 id="infogain"> Information Gain </h6>
	Information gain is indirectly proportional to the probability of the event. An event having a probability of occurrence 1 has no information at all (it is certain to happen). An event having 0 probability of occurrence contains the most information.
	
	![](https://render.githubusercontent.com/render/math?math=I(x)%20=%20-log%20P(x)&mode=display)
	
	Information gained when an unfair coin is tossed is less than information gained on a fair coin, since unfair coin's outcome is the unfair side most of the times. 

*	<h6 id="surprise"> Surprise Factor </h6>
	The surprise factor is directly proportional to information gain. The most probable event will have the least surprise factor (eg: sun rises in the east). Least probable event will have the most surprise factor (eg: sudden death of xyz)
	

<h3 id="xentropy"> Cross Entropy </h3>
---

The expected amount of information gained when a scheme optimised for one distribution is applied to another distribution is quantified by cross-entropy.

Amount of information gained when you think I'm tossing a fair coin but secretly, I'm tossing an unfair coin is given by ![](https://render.githubusercontent.com/render/math?math=H(P_{unfair},P_{fair})%20=%20-%5Cmathbb{E}_{x~P_{unfair}}%20log%20P_{fair}(x)&mode=display)

On the other hand, amount of information gained when you think I'm tossing an unfair coin but secretly, I'm tossing a fair coin is given by ![](https://render.githubusercontent.com/render/math?math=H(P_{fair},P_{unfair})%20=%20-%5Cmathbb{E}_{x~P_{fair}}%20log%20P_{unfair}(x)&mode=display)

In any scenario ![](https://render.githubusercontent.com/render/math?math=H(P_{fair},P_{unfair})%20%3E%20H(P_{unfair},P_{fair})&mode=display),
because whenever the unfair coin comes up with anything other than the unfair side, you're pretty surprised. But when I toss the fair coin, it comes up something other than unfair side most of the time -- so if you think I'm tossing the unfair coin but I'm not, you're pretty surprised most of the time!


<h3 id="kldivergence"> Kullback - Leiber divergence </h3>
---

The penalty charged when one optimization scheme is used on other distribution is quantified by KL divergence
![](https://render.githubusercontent.com/render/math?math=D_{KL}(P||Q)=%20%5Cmathbb{E}_{x~P}%20[%5Clog%20%5Cfrac{P(x)}{Q(x)}]%20=%20%20%5Cmathbb{E}_{x~P}[%5Clog%20P(x)%20-%20%5Clog%20Q(x)])

In other words, the extra information gained when I toss a fair coin but you mistakenly believe I'm tossing an unfair coin than if I toss the fair coin and you correctly believe I'm doing so.

<h3 id="facts"> Facts about KL divergence </h3>
---

* KL divergence is non-negative
* KL divergence is 0 if and only if P and Q are of the same distribution (incase of discrete variables), or equal "almost everywhere" (incase of continuous variables)
* It is often conceptualized as measuring distance between distributions, but it is not actually a distance measure since it doesn't follow the triangle law (not commutative)


<h3 id="usage"> Usage in Machine Learning </h3>
---

1. KL divergence is used in ML to measure the information loss in the fitted model relative to that in the reference model
2. It is widely used in variational inference, where an optimization problem is constructed that aims at minimizing the KL-divergence between the intractable target distribution `P` and a sought element `Q` from a class of tractable distributions.

<h3 id="references"> References </h3>
---
*   [https://www.quora.com/What-is-a-good-laymans-explanation-for-the-Kullback-Leibler-divergence](https://www.quora.com/What-is-a-good-laymans-explanation-for-the-Kullback-Leibler-divergence)
*   [http://www.deeplearningbook.org/contents/prob.html](http://www.deeplearningbook.org/contents/prob.html)