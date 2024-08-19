<img src = 'img/trees.png'></img>

## Boosting 

### Adaboost

Boosting adds a set of weights, to a given set of samples, to iteratively train a set of ensembled trees, allowing them to pay more attention to classes that they've incorrectly classify in previous iterations, to then correctly classify them in further iterations

**Given a set of training examples, you**

1. Train a weak learner, a model that is only slightly better than random guessing.
2. Identify the misclassified and correctly classified classes. 
3. Increase the weights for the samples in the misclassified class
4. Retrain the model on the new set of samples, by applying the new weights to compute the amount of say $(\alpha)$.
5. For a final prediction, run a pass through all the weak learners, taking into account their predictions multiplied by $\alpha$. If the final output is positive, the prediction is $1$. Otherwise, the prediction is $-1$ (or $0$ depending on what your labels are).

Say we have a boosted classifier, $H(\vec{x}) = \sum_{i=1}^T \alpha_t h_t(\vec{x})$ where $h_t$ is an instance of a **weak learner**, a model / algorithm that is not good at classifying predictions. This $h_t$ is typically a tree stump, a decision tree that has a depth $< 1$, classifying it as a weak learner.

The weak learner, $h_t$, is trained on a given dataset, only allowed to reach $depth = 1$. The total error for the stump is $\frac{\sum_{i:h(x_i)≠y_i w_i}}{\sum w}$, where $w_i$ is the $ith$ weight associated with the $ith$ classifier that provided an incorrect prediction.

- This total error will always be within the range $[0, 1]$, as the weights are always normalized to sum up to $1$, per $\sum w$.
- This total error ($\epsilon$) will determine the amount of say or contribution, $\alpha$, that a model has on the final output of the ensemble.
- If $\epsilon > .5$ then $\epsilon = 1 - \epsilon$ and the prediction of the stump is then flipped to an opposite class label, $-1$ (or $0$ in some cases). This is as weak learners must perform better than random guesses, which random guesses have a $50$% chance of being correct.

The amount of say for a tree $\alpha$ determines the total contribution that a stump has into the entire ensemble. During testing, the output of a stump, $h_t$, is multiplied by $\alpha$ (as $h_t \cdot \alpha$). This scales down the output of the model by the amount of say, determining how much of a contribution the predictor, $h_t$ has into the entire ensemble.

$\alpha$ is computed as,

$\alpha = \frac{1}{2} ln(\frac{1 - \epsilon}{\epsilon})$

where $\epsilon$ is the total error for a given $h_t$.

When $\epsilon$ is small, the amount of say will be large. Otherwise, the amount of say will be small.

As an example, if the total error was defined as $\frac{5}{8}$ or $.625$, the amount of say would be calculated as:

$\epsilon = 1 - \epsilon = .375$

$\alpha = ln (\frac{(1-.375)}{.375}) * (\frac{1}{2})$

$\alpha = .8\overline{33}$

and note that in this case, the $\epsilon$ was $> .5$, therefore the original default prediction of the stump as $1$ was flipped to the opposite class $-1$ or $0$. 

You can also compute the loss function of the entire ensemble $H$ by using the *exponential loss* as:

$loss_{exp} = e^{-y \cdot \hat{y}}$

where $y$ are the true labels and $\hat{y}$ are the predictions of the entire ensemble.

> [!NOTE]

The predictions of the ensemble are initially computed as the sum of all raw predictions of the $ith$ stump multiplied by the corresponding $\alpha$ of the $ith$ stump. Lets call this $\hat{y}_{raw}$.

The final predictions of the ensemble are then computed as:

$\hat{y} = sign(\hat{y}_{raw})$

where the output $\hat{y}$ will either be $-1$ or $1$ (which can be modified to output $0$ by using `np.where` instead of `np.sign` in practical implementations).

<div align = 'center'>

```
y_hat = np.where(y_raw > 0, 1, 0)
```
</div>

> END OF NOTE

Once $\epsilon$, $\alpha$, and the $loss_{exp}$ are computed, for every incorrect mapping that each $h_t$ applies on $\vec{x} \rightarrow \vec{y}$, where $\vec{y}$ are the true labels, you take the initial weak learner $h_1$, and iteratively update the weights $(w)$ on the set of samples, such that the learner $h_1$ pays more attention to the weighted samples at it's second iteration. 

Each new stump pays more attention to misclassified samples through the use of $w$ to compute the $\epsilon$ which is then used to compute $\alpha$.

We can update $w$ as:

$w = w(e^{-\vec{\alpha} \vec{h(x_i)}y_i})$

where $y_i$ is the true label for the training sample $x_i$ and $\alpha$ is the amount of say that each ensemble, $\vec{h(x_i)}$ is the label prediction from the weak learners.

Then the weights, $w$, are normalized such that the sum of all weights $w$ add up to $1$. 

This can be done by: 

$w_{sum} = \sum w$ <br>
$w = \frac{w}{w_{sum}}$ <br>

This is done until the total number of specified stumps are trained. Then, the final model consists of all weak learners, with the updated weights, attained at the final iteration of the training. 

Each weak learner then makes a final prediction, based on their amount of say, contributing to the final ensemble as:

$H(\vec{x}) = \sum\alpha h(\vec{t})$

If $H(\vec{x}) > 0$, the sample is classified as $1$, otherwise the sample is classified as $0$ or $-1$, depending on what the opposing label is identified as.

> Again, this can be done through the sign() function.

### Gradient Boosting

Gradient boosting is analogous to AdaBoost, but it's specifics differ.

- It fits decision trees sequentially to the errors of previous trees
- Each tree that is fit is deeper than the previous tree, fitting weak learners to reach a strong learner
- It's loss function is differentiable, allowing us to compute gradients to train each subsequent tree.

1. First, you construct a base tree (the root node)
2. Compute the errors and predictions for the current tree.
3. Then you build a new tree based on the errors of the current tree.
4. You combine the current tree and the new tree and then repeat fom step $2$.

**Regression**

Say we denote our classifiers as $h_i$ where $i$ is the $ith$ classifier.

For the root node, given a training set $X$ and labels $Y$, the prediction is based upon the average of $Y$, $\hat{Y} = \frac{\sum_{i=1}^n y_i}{n}$ You can compute the loss of the initial prediction as the mean squared error (mse): $mse = \frac{1}{n} \sum_{i = 1}^{n} y - \hat{y_i}$

Then the next tree is built based on the errors of the previous tree, in this case, a root node.

First compute the residuals:

$R = Y - \hat{Y}$

and then train a new tree to fit on these residuals rather than the labels. $r_i$ is the new $y_i$.

> *Note that regression trees are fit and their nodes are split, based on the Mean Squared Error or Mean Absolute Error.*

For each leaf node in a new tree, the value at the leaf node (label for a correpsonding sample) is the new residual $r$. 

If a tree has multiple values at a leaf node, then the residual is the average of all the values. The samples at the given leaf node with non-unique values, then have their residuals assigned as that same average value. Both samples would get transfered the same residual via a weighted sum (below).

So to transfer the new residuals, to the previous, you can take a weighted sum of the previous residuals with the new residuals as:

$R = R^{t-1}  + \alpha R^t$

where the residuals, $R$, become the predictions of the current tree: $\hat{Y} = R$.

Here $\alpha$ is a hyperparameter one can tune, the learning rate.

Then essentially, the process at each step is:

$\hat{Y} = \frac{1}{n} \sum_{i = 1}^{N} L(y_i, h(x_i + \hat{y}_{t-1}))$

$h_t(x) = h_{t-1}(x) + \alpha\sum\hat{Y}$

$repeat \hspace{1mm} t \hspace{1mm} times$

> We're trying to get the prediction that minimizing the value of the loss function, based on labels Y and the predictions of the algorithm.

**Classification**

For classification, the process is similar, with some subtle differences.

For the root node, the prediction is based on the log odds of $Y$. Say the prediction is $\hat{Y}$.

$\hat{Y}_{raw} = \log(\frac{p}{1-p})$

where $p$ is the probability of a given $Y$.

Then, $\hat{Y}_{raw}$ becomes the initial value for the initial tree's leaf nodes. Then $\hat{Y}_{raw} is transformed into a probability, using the sigmoid $(\sigma)$ or softmax function, commonly used in neural networks, depending if you're doing binary or multiclass classification:

$Sigmoid = \frac{1}{1 + e^{-z}} = \hat{Y}$ where $z = \hat{Y}_{raw}$ 

$Softmax = \frac{e^{z_k}}{\sum e^{z_j}} = \hat{Y}$ where $z = \hat{Y}_{raw}$ 

You can then compute the loss using the categorical cross-entropy loss:

$L = - \sum_k Y_k \log(\hat{Y}_k)$, for softmax (multiclass)

$L = - \sum [Y \log(\hat{Y}) + (1 - Y) \log(1-\hat{Y})]$ for sigmoid (binary)

You then compute the residual / gradient of the loss with respect to the prediction:

- For binary classification:
$R = Y - \sigma(\hat{Y}_{raw})$

- For multiclass classification:
$R = Y_k - \frac{e^{\hat{Y}_{raw}}}{\sum e^{\hat{Y}_{raw}}}$

These residuals will then be used to grow the next tree.

We can directly use these residuals to fit the next tree, as they represent the direction and magnitude of change needed in the log-odds space.

We update the boosted ensemble as:

$\hat{Y}_{raw} = \hat{Y}_{raw}^{t-1} + \alpha \cdot h_t(x)$

where $\alpha$ is a learning rate and $h_t(x)$ is the prediction of the new tree, with respect to the residuals.

This process is repeated for a specified number of iterations or until a stopping criterion is met.

The final prediction is obtained by transforming the final $\hat{Y}_{raw}$ to probabilities using sigmoid or softmax functions. 

In softmax, for a given class, the label with the highest probability is the assigned class for the sample
In sigmoid, if the output is above $.5$, then the class is $1$, otherwise it is $0$
