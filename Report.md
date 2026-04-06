In this work, two forecasting formulations are compared: a **single-output approach** and a **multi-output approach**. Although both pursue the same overall goal of predicting electricity prices for BESS optimization, they differ not only in the target structure, but also in the way the models are trained.

In the **single-output setup**, each model is trained to predict only **one target value** at a time, for example the electricity price of one specific hour. Thus, one sample corresponds to one forecast target, and the model learns a direct mapping from the input features to this single output.

In the **multi-output setup**, the target of each sample is the full **24-hour price vector** of one day. However, because the implementation uses `MultiOutputRegressor`, this does **not** mean that one single model jointly learns all 24 hours together. Instead, the wrapper trains **24 separate models in parallel**, one for each output hour. In other words, the model structure is multi-output from the user perspective, but internally it is decomposed into 24 independent prediction tasks:

$$
\hat{\mathbf{Y}}_d =
\left[
\hat{Y}_d^{1},
\hat{Y}_d^{2},
\dots,
\hat{Y}_d^{24}
\right]
$$

with one model fitted for each component $$\hat{Y}_d^{h}$$.

This has an important implication for the comparison with a single-output setup. In the single-output case, one model predicts one hour. In the present multi-output implementation, 24 separate models are trained, all using the same daily feature matrix, but each focusing on a different forecast hour. Therefore, the method is not a native joint multi-output model, but rather a collection of parallel single-output models organized under a multi-output interface.

As a result, differences between the two approaches arise from several sources: the sample definition, the target structure, the feature representation, and the training strategy. The single-output setup uses one model for one target, whereas the current multi-output setup uses **24 parallel models** for the 24-hour forecast horizon. Therefore, model performance should not be compared only at the algorithm level, but also in view of these structural differences.



In a **single-output formulation**, each training sample is associated with exactly one target value. Denoting the feature vector at sample $$i$$ by $$\mathbf{x}_i$$ and the corresponding target by $$y_i$$, the learning task can be written as

$$
\hat{y}_i = f(\mathbf{x}_i),
\qquad
y_i \in \mathbb{R}.
$$

Thus, one model learns one mapping from the input space to a scalar output:

$$
f:\mathbb{R}^p \rightarrow \mathbb{R}.
$$

In contrast, in the present **daily 24-hour forecasting setup**, one sample corresponds to one day $$d$$, and the target is the full vector of 24 hourly prices:

$$
\mathbf{Y}_d =
\begin{bmatrix}
Y_d^{1} \\
Y_d^{2} \\
\vdots \\
Y_d^{24}
\end{bmatrix}
\in \mathbb{R}^{24}.
$$

Accordingly, the prediction problem is written as

$$
\hat{\mathbf{Y}}_d = F(\mathbf{X}_d),
\qquad
\hat{\mathbf{Y}}_d \in \mathbb{R}^{24}.
$$

At first sight, this appears to be one model learning a mapping

$$
F:\mathbb{R}^p \rightarrow \mathbb{R}^{24}.
$$

However, this is **not** what the current implementation does. Because the setup uses `MultiOutputRegressor`, the 24-dimensional target is decomposed into **24 separate regression problems**. More precisely, for each forecast hour $$h \in \{1,\dots,24\}$$, one independent model $$f_h$$ is trained:

$$
\hat{Y}_d^{h} = f_h(\mathbf{X}_d),
\qquad h = 1,\dots,24.
$$

So instead of one native joint multi-output model, the current approach fits the collection

$$
\{f_1, f_2, \dots, f_{24}\}.
$$

The final daily prediction vector is then assembled as

$$
\hat{\mathbf{Y}}_d =
\begin{bmatrix}
f_1(\mathbf{X}_d) \\
f_2(\mathbf{X}_d) \\
\vdots \\
f_{24}(\mathbf{X}_d)
\end{bmatrix}.
$$

Therefore, the present “multi-output” setup is operationally better described as **24 parallel single-output models** that share the same day-level input matrix $$\mathbf{X}_d$$, but are estimated separately for each output hour.

This differs from Cecilia’s single-output setup in two ways. First, her formulation predicts only one target at a time:

$$
\hat{y}_t = f(\mathbf{x}_t),
$$

where $$t$$ denotes one hourly timestamp. Second, only one model is trained for this scalar target. In the present implementation, by contrast, one daily sample is used to estimate 24 separate hourly targets, which means that 24 models are trained in parallel.

Hence, the comparison is not only a comparison of algorithms, but also of modeling structure:

$$
\text{Single-output: } \mathbb{R}^p \rightarrow \mathbb{R}
$$

versus

$$
\text{Current multi-output implementation: } 
\mathbb{R}^p \rightarrow \mathbb{R}^{24}
\quad \text{via 24 separate mappings } f_h:\mathbb{R}^p \rightarrow \mathbb{R}.
$$

This distinction is important, because differences in performance can arise not only from the model class itself, but also from the different target definition, sample construction, feature representation, and training strategy.