TODO: REVIEW NOTES

## Boosting 

### Adaboost

Boosting adds a set of weights, to a given set of samples, to iteratively train a set of ensembled trees, allowing them to pay more attention to classes that they've incorrectly classify in previous iterations.

Given a set of training examples, you

1. Train a weak learner, meaning a model that is only slightly better than random guessing.
2. Identify the misclassified and correctly classified classes. 
3. Increase the weights for the samples in the misclassified class (can be done via increaisng probability of drawing a class in bootstrapping)
4. Retrain the model on the new set of samples.
5. For a final prediction, run a pass through all the weak learners and 

Say we have a boosted classifier:
$H(\vec{x}) = \sum_{i=1}^T \alpha_t h_t(\vec{x})$

$h_t$ is an instance of a **weak learner**, a model / algorithm that is not good at classifying predictions.

This is typically a tree stump, a decision tree that has a depth $< 1$.

The weak learner, $h_t$, is trained on a given dataset, only allowed to reach $depth = 1$. The total error for the stump is the $\sum_{i:h(x_i)≠y_i} w_i$, where $w_i$ is the $ith$ weight associated with the $ith$ classifier that provided an incorrect prediction.

This total error will always be within the range $[0, 1]$, as the weights are always normalized to sum up to $1$.

This total error ($\epsilon$) will determine the amount of say or contribution, $\alpha$, that a model has on the final output of the ensemble.

The $\alpha$ is computed as:

$\alpha = \frac{1}{2} ln(\frac{1 - \epsilon}{\epsilon})$

When the total error is small, the amount of say will be large and positive. Otherwise it will be a large negative value.

Then, if $\alpha$ is a negative value, the learner's predictions are incorrect and what would've been, for example, a prediction of $1$ for a positive value, will be turnt into a value representing the opposite class, typically $-1$ or $0$.

As an example, if the total error was defined as $\frac{3}{8}$ or $.625$, the amount of say would be calculated as:

$\alpha = ln (\frac{(1-.625)}{.625}) * (\frac{1}{2})$

For every incorrect mapping that each $h_t$ applies on $\vec{x} \rightarrow \vec{y}$, where $\vec{y}$ are the true labels, you take the initial weak learner $h_1$, and iteratively update the weights on the set of samples, such that the learner $h_1$ pays more attention to the weighted samples at it's second iteration. 

Once the loss function is computed, the weight update can be computed as:

$w \leftarrow w(e^{-\vec{\alpha} \vec{h(x_i)}y_i})$

*This equation can be eperated into 2 equations as:*

*If correct: $w \leftarrow w(e^{-\vec{\alpha}})$* <br>
*Else: $w \leftarrow w(e^{\vec{\alpha}})$*

*The first more explicitly computes the full weight increase / decrease operation in a single equation while the latter does it seperately.*

where $y_i$ is the true label for the training sample $x_i$ and $\alpha$ is the amount of say that each ensemble, $\vec{h(x_i)}$ is the label prediction from the weak learners.

Then the weights, $w$, are normalized such that the sum of all weights $w$ add up to $1$. 

This can be done by: 

$w_{sum} = \sum w$ <br>
$w_{normalized} = \frac{w}{w_{sum}}$ <br>
$w \leftarrow w_{normalized}$

We can now compute the loss (error metric) of the entire model as:

$l = \sum_{i=1}^ne^{-yh(x_i)_i}$

O
Then we train another instance of the algorithm, $h_2$, applying the weights $w$. This can be done via weighted bootstrapping or a weighted gini index / entropy.

- The weighted gini index would look as $1 - \sum w^2$, replacing $w$ with the original probability $p$.
- Taking a new dataset, based on weighted bootstrapping, would just increase the probability that a given sample is drawn based on the weights. You draw a new dataset baesd on the weights, but then reset the weights each time, allowing the weak learner to generate new weights on the weighted boostrapped dataset.

The final model consists of all weak learners, with the updated weights, attained at the final iteration of the training. 

Each weak learner then makes a final prediction, based on their amount of say, contributing to the final ensemble as:

$H(\vec{x}) = \sum\alpha h(\vec{t})$

If $H(\vec{x}) > 0$, the sample is classified as $1$, otherwise the sample is classified as $0$ or $-1$, depending on what the opposing label is identified as.

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

For the root node, given a training set $X$ and labels $Y$, the prediction is based upon the average of $Y$:

$\hat{Y} = \frac{\sum_{i=1}^n y_i}{n}$

You can compute the loss as the mean squared error (mse).

$mse = \frac{1}{n} \sum_{i = 1}^{n} y - \hat{y_i}$

Then we build the next tree based on the errors of the previous tree, in this case, a root node.

First compute the residuals:

$R = Y - \hat{Y}$

and then train a new tree to fit on these residuals rather than the labels. $r_i$ is the new $y_i$.

> *Note that regression trees are fit and their nodes are split, based on the Mean Squared Error or Mean Absolute Error.*

For each leaf node in a new tree, the value at the leaf node (label for a correpsonding sample) is the new residual $r$. 

If a tree has multiple values at a leaf node, then the residual is the average of all the values. The samples at the given leaf node with non-unique values, then have their residuals assigned as that same average value. Both samples would get transfered the same residual via a weighted sum (below).

So to transfer the new residuals, to the previous, you can take a weighted sum of the previous residuals with the new residuals as:

$R = R^{t-1}  + \alpha R^t$

where the residuals, $R$, becoems the predictions of the current tree.

$\hat{Y} = R$

where $\alpha$ is defined as the learning rate.

Then essentially, the process at each step is:

$\hat{Y} = \frac{1}{n} \sum_{i = 1}^{N} L(y_i, h(x_i + \hat{y}_{t-1}))$

$h_t(x) = h_{t-1}(x) + \alpha\sum\hat{Y}$

$repeat \hspace{1mm} t \hspace{1mm} times$

> We're trying to get the prediction that minimizing the value of the loss function, based on labels Y and the predictions of the algorithm.

**Classification**

For classification, the process is similar, with some subtle differences:

For the root node, the prediction is based on the log odds of $Y$. Say the prediction is $\hat{Y}$.

$\hat{Y}_{raw} = \log(\frac{p}{1-p})$

where $p$ is the probability of a given $Y$.

Then, $\hat{Y}_{raw}$ becomes the initial value for the initial tree's leaf nodes.

Then you transform it into a probability, using the sigmoid $(\sigma)$ or softmax function, commonly used in neural networks, depending if you're doing binary or multiclass classification:

$Sigmoid = \frac{1}{1 + e^{-z}} = \hat{Y}$ where $z = \hat{Y}_{raw}$ 

$softmax = \frac{e^{z_k}}{\sum e^{z_j}} = \hat{Y}_k$ where $z = \hat{Y}_{raw}$ 

You can then compute the loss using the categorical cross-entropy loss:

$L = - \sum_k Y_k \log(\hat{Y}_k)$, for softmax (multiclass)

$L = - \sum [Y \log(\hat{Y}) + (1 - Y) \log(1-\hat{Y})]$ for sigmoid (binary)

You then compute the residual / gradient of the loss with respect to the prediction:

For binary classification:
$R = Y - \sigma(\hat{Y}_{raw})$

For multiclass classification:
$R_k = Y_k - \frac{e^{\hat{Y}_{raw,k}}}{\sum_j e^{\hat{Y}_{raw,j}}}$

These residuals will then be used to grow the next tree.

We can directly use these residuals to fit the next tree, as they represent the direction and magnitude of change needed in the log-odds space.

We update the boosted ensemble as:

$\hat{Y}_{raw} = \hat{Y}_{raw}^{t-1} + \alpha \cdot h_t(x)$

where $\alpha$ is a learning rate and $h_t(x)$ is the prediction of the new tree, with respect to the residuals.

This process is repeated for a specified number of iterations or until a stopping criterion is met.

The final prediction is obtained by transforming the final $\hat{Y}_{raw}$ to probabilities using sigmoid or softmax functions. 

In softmax, for a given class, the label with the highest probability is the assigned class for the sample
In sigmoid, if the output is above $.5$, then the class is $1$, otherwise it is $0$
