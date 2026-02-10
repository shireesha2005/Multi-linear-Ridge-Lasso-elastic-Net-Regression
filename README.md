
Implementation of Multi-linear regression,Ridge Regression, Lasso Regression, Elastics Net Regression with real-world dataset(House Pricing)

Linear regression loss

 L = 1/n sumation of (y - y cap)^2
No additional term is added.
The loss only depends on the prediction error (Mean Squared Error).

What does it encourage or discourage?
Encourages the model to fit the training data as closely as possible.Does not discourage large coefficients.

Effect on model complexity and generalization
Can result in large weights when features are correlated.High risk of overfitting, especially with many features.Generalization may be poor when the dataset is noisy or high-dimensional.

Ridge Regression Loss
What term is added?
L2 penalty: the sum of squared coefficients.

What does it encourage or discourage?
Discourages large coefficient values.Encourages weights to be small and evenly distributed.

Effect on model complexity and generalization
Reduces model complexity by shrinking coefficients.Handles multicollinearity well.Improves generalization by preventing overfitting.Does not perform feature selection (coefficients rarely become zero).

Lasso Regression Loss
What term is added?
L1 penalty: the sum of absolute values of coefficients.

What does it encourage or discourage?
Strongly discourages unnecessary features.Encourages sparsity by forcing some coefficients to become exactly zero.

Effect on model complexity and generalization
Performs automatic feature selection.Reduces model complexity significantly.Works well when only a few features are truly important.Can be unstable when features are highly correlated.

Elastic Net Loss
What term is added?
A combination of L1 and L2 penalties.

What does it encourage or discourage?
Encourages sparsity (like Lasso).Discourages large coefficients (like Ridge).Groups correlated features together.Effect on model complexity and generalizationBalances feature selection and stability.Handles correlated features better than Lasso.Produces a more robust and generalizable model.


Why does regularization improve test performance?
From the training vs testing error plots, we observe:Linear Regression achieves very low training error But its test error is significantly higher.This indicates overfitting — the model learns noise and overly complex patterns in training data.
Regularization improves test performance because:
It penalizes large coefficients

Reduces model complexity

Prevents the model from fitting noise

Encourages smoother, more generalizable solutions

Why does Ridge keep all features but shrink them?

From the coefficient shrinkage path plot for Ridge:
All coefficients gradually move towards zero
None of the coefficients become exactly zero
This happens because Ridge uses L2 regularization, which:
Penalizes the square of coefficients
Makes large weights very costly
Encourages small but non-zero weights
