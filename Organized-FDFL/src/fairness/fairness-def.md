Individual Fairness (f₁) in Fair Regression
$$f_{1}(w) = \frac{1}{n_{0}n_{1}} \sum_{i \in G_{0}} \sum_{j \in G_{1}} d(y_i, y_j),\left( w\cdot x_i - w\cdot x_j \right)^2,$$
$$\frac{\partial f_{1}(w)}{\partial w} = \frac{2}{n_{0}n_{1}} \sum_{i \in G_{0}} \sum_{j \in G_{1}} d(y_i, y_j),\left( w\cdot x_i - w\cdot x_j \right),\left( x_i - x_j \right).$$
Group Fairness (f₂) in Fair Regression
$$f_{2}(w) = \left( \frac{1}{n_{0}n_{1}} \sum_{i \in G_{0}} \sum_{j \in G_{1}} d(y_i, y_j),\left( w\cdot x_i - w\cdot x_j \right) \right)^2,$$
$$\frac{\partial f_{2}(w)}{\partial w} = \frac{2}{n_{0}n_{1}} \left( \sum_{i \in G_{0}} \sum_{j \in G_{1}} d(y_i, y_j),\left( w\cdot x_i - w\cdot x_j \right) \right) \left( \sum_{i \in G_{0}} \sum_{j \in G_{1}} d(y_i, y_j),\left( x_i - x_j \right) \right).$$
Atkinson's Inequality Index and Gradient
$$b_i = \left(\hat{y}i - y_i\right)^2, \quad \mu = \frac{1}{n} \sum{i=1}^{n} b_i,$$
For $\beta \neq 1$:
$$A_\beta = 1 - \frac{1}{\mu} \left( \frac{1}{n} \sum_{i=1}^{n} b_i^{,1-\beta} \right)^{\frac{1}{,1-\beta}},$$
For $\beta = 1$:
$$A_1 = 1 - \frac{1}{\mu} \left( \prod_{i=1}^{n} b_i \right)^{\frac{1}{n}}.$$
The gradient for the $k$-th element (assuming $\beta \neq 1$):
$$\frac{\partial A_\beta}{\partial \hat{y}_k} = -\frac{2(\hat{y}_k - y_k)}{n} \left[ \frac{U^{\frac{\beta}{,1-\beta}}}{(1-\beta),\mu},b_k^{-\beta} - \frac{U^{\frac{1}{1-\beta}}}{\mu^2} \right],$$
with
$$U = \frac{1}{n}\sum_{i=1}^{n} b_i^{,1-\beta}.$$
Bounded Statistical Parity and Gradient
$$\overline{\hat{y}}0 = \frac{1}{n_0}\sum{i \in G_0}\hat{y}_i, \quad \overline{\hat{y}}1 = \frac{1}{n_1}\sum{j \in G_1}\hat{y}_j,$$
$$\overline{y}0 = \frac{1}{n_0}\sum{i \in G_0}y_i, \quad \overline{y}1 = \frac{1}{n_1}\sum{j \in G_1}y_j,$$
$$f_{\text{SP}} = \max\left{0,; |\overline{\hat{y}}_0-\overline{\hat{y}}_1| - \left(|\overline{y}_0-\overline{y}_1|+\varepsilon\right) \right}.$$
If
$$|\overline{\hat{y}}_0-\overline{\hat{y}}_1| > |\overline{y}_0-\overline{y}_1|+\varepsilon,$$
then the gradients are:
$$\frac{\partial f_{\text{SP}}}{\partial \hat{y}_i} = \frac{\operatorname{sgn}(\overline{\hat{y}}_0 - \overline{\hat{y}}_1)}{n_0},\quad i \in G_0,$$
$$\frac{\partial f_{\text{SP}}}{\partial \hat{y}_j} = -\frac{\operatorname{sgn}(\overline{\hat{y}}_0 - \overline{\hat{y}}_1)}{n_1},\quad j \in G_1.$$
If the penalty is zero, the gradients are zero.
