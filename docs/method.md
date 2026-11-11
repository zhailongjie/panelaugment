# Method: fixed-panel conditional discriminative augmentation

PA receives normalized training expression $X\in\mathbb{R}^{N\times P}$, class labels $y_i$, group identifiers $g_i$, an ordered candidate-gene axis, an ordered base $B$, and an additional budget $k$. It returns an ordered panel whose first $|B|$ entries are exactly $B$, followed by $k$ distinct candidates outside the base.

Groups refer to the study's prespecified biological or sample units. A group does not necessarily correspond to an independent animal when libraries are pooled or linked. Estimation of the selector statistics below uses the selection-training role only. The preceding [PA data preprocessing](preprocessing.md) combines training-fitted candidate QC with measurement-consistency gating on the declared donor aggregates; those measurement inputs have their own recorded data provenance.

## Balanced training moments

Let $C$ denote the number of classes, $G_a$ the groups in which class $a$ appears, and $n_{ag}$ the number of cells in observed group × class combination $(a,g)$. Each such cell receives weight

$$
w_i=\frac{1}{C\,|G_a|\,n_{ag}},\qquad \sum_i w_i=1.
$$

Classes have equal total mass; observed groups have equal mass within each class. Absent combinations receive no observations or weights.

Compute the weighted mean and population variance of each gene:

$$
\mu_j=\sum_i w_iX_{ij},\qquad
v_j=\sum_iw_i(X_{ij}-\mu_j)^2,\qquad
s_j=\sqrt{\max(v_j,10^{-8})}.
$$

Set $Z_{ij}=X_{ij}/s_j$. The global mean is used to calculate variance and is **not** subtracted from $Z$. For each observed group × class combination, calculate its unweighted mean $\mu_{ag}$ in the $Z$ coordinates and residuals $R_i=Z_i-\mu_{y_i g_i}$. The common within-class residual covariance is

$$
\Sigma=\sum_i w_iR_iR_i^\top.
$$

The implementation symmetrizes this matrix as $(\Sigma+\Sigma^\top)/2$. It does not apply an $n-1$ or effective-degrees-of-freedom correction.

For class pair $a\lt b$, let $H_{ab}=G_a\cap G_b$. Its contrast is

$$
\delta_{ab}=\frac{1}{|H_{ab}|}\sum_{g\in H_{ab}}(\mu_{ag}-\mu_{bg}).
$$

An empty $H_{ab}$ is an error. Averaging the two classes over separate group sets before subtracting would be a different estimator. Pair-support metadata records the shared groups and their count. Equal weighting does not remove uncertainty caused by few groups or few cells.

## Objective and exact conditional gain

With positive ridge $\lambda$, define $A=\Sigma+\lambda I$. The manuscript setting is $\lambda=1$; regularization is applied **after** training scaling. For selected indices $S$,

$$
d_{ab}(S)=\delta_{ab,S}^{\top}A_{SS}^{-1}\delta_{ab,S},\qquad
F(S)=\sum_{a\lt b}\log(1+d_{ab}(S)).
$$

All unordered class pairs have equal weight in this sum. For candidate $j\notin S$, the Schur complement and distance increment are

$$
\tau_j=A_{jj}-A_{jS}A_{SS}^{-1}A_{Sj},
$$

$$
q_{ab}(j\mid S)=
\frac{(\delta_{ab,j}-A_{jS}A_{SS}^{-1}\delta_{ab,S})^2}{\tau_j}.
$$

The ranking score is the exact marginal objective gain

$$
\Delta F(j\mid S)=\sum_{a\lt b}\log\left(1+\frac{q_{ab}(j\mid S)}{1+d_{ab}(S)}\right).
$$

The log transformation reduces the gain assigned to a fixed distance increment for an already well-separated class pair. This is a design property, not evidence that any particular test class improves.

## Greedy search and numerical updates

The base is incorporated in its supplied order. Remaining steps choose the available candidate with greatest gain. Ties use the first candidate in the fixed axis, as with `numpy.argmax`. The procedure has no random sampling.

The implementation maintains conditional contrasts $e$, conditional variances $v$, accumulated pair distances $d$, and incremental Cholesky columns $L$. Initially $e=\delta$, $v=\mathrm{diag}(A)$, $d=0$. Adding gene $j$ at step $t$ gives

$$
\ell=(A_{:j}-L_tL_t[j,:]^\top)/\sqrt{v_j},\qquad
z_{ab}=e_{ab,j}/\sqrt{v_j},
$$

$$
d_{ab}\leftarrow d_{ab}+z_{ab}^2,\quad
e_{ab}\leftarrow e_{ab}-z_{ab}\ell^\top,\quad
v\leftarrow v-\ell^2,\quad
L_{t+1}=[L_t,\ell].
$$

For valid training moments, covariance is positive semidefinite in exact arithmetic; positive ridge makes $A$ positive definite. Available candidates must have positive conditional variance. Candidate-score division uses a $10^{-15}$ lower guard, while the chosen update uses its actual variance. Nonfinite values and invalid numerical states are errors; the software does not silently tune ridge to conceal them.

The trace records selected indices, compulsory-base status, objective, total gain, and per-pair distance increments. It includes base-incorporation steps, which must not be interpreted as optional greedy choices.

For fixed training moments, candidate order, ordered base and parameters, larger-budget searches share prefixes with smaller-budget searches. `SelectionResult.prefix` exposes this property. A different base needs a new search; a different training fold needs new statistics and selection.

`diagonal=True` replaces $\Sigma$ by its diagonal before adding ridge. It leaves contrasts, base, and objective unchanged and serves as a covariance ablation.

## Normalization is part of the input definition

The selector expects normalized expression. `log_normalize` implements

$$
\log\left(1+\alpha\frac{x_{ij}}{\max(D_i,1)}\right),
$$

where $\alpha$ is `target_sum` (default `10000`). Provide $D_i$ measured across the intended full library when input columns contain only candidates. With omitted `library_totals`, the input matrix's row sums define $D_i$. Nonnegative all-zero rows remain zero. The caller must establish that the denominator matches the intended experiment.

Selection, classifier evaluation and structure evaluation may use different normalization definitions. The manuscript's shared classifier uses the current panel's denominator, whereas selection uses full-library totals and structure evaluation uses the common candidate denominator. This toolkit does not implement those complete evaluation pipelines or merge their normalization conventions.

## Computational cost

Let $Q=C(C-1)/2$, $T=|B|+k$, and $H$ be the number of observed group × class combinations. Dense covariance multiplication costs $O(NP^2)$. Search costs $O(QPT+PT^2)$, including compulsory base steps, because the frozen update computes candidate gains at those steps too.

Dominant numeric storage is $O(NP+P^2+QP+PT)$: dense expression/residual arrays, covariance, contrasts and factors. Sparse input support does not imply a sparse covariance algorithm. Group bookkeeping and pair-support metadata add costs; group masks can require $O(NH)$ work, and shared-group contrasts up to $O(QGP)$. Wall-clock timing must include moment estimation when describing end-to-end performance.

## Claims and limitations

The fixed-statistics set objective is monotone in exact arithmetic. General submodularity, a global optimum, and a $1-1/e$ approximation guarantee are not asserted. The objective is not a classifier loss or exact conditional mutual information. An improved training objective does not ensure higher external balanced accuracy, F1, class recall, or expression-neighborhood preservation.

PA relies on training labels, shared-group support, candidate coverage, and covariance estimation. Probe feasibility and unknown-state discovery are outside this objective. Keeping a preselected set and choosing complementary additions are established capabilities in existing panel-design methods; the package makes no priority claim for those general ideas.
