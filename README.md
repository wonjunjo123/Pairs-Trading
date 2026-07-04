
# Documentation / Choices

Made sure to eliminated lookahead bias 

Screening pipeline
But there is always posssibility of type I error since we are screening many pairs.
screening.py screens every within-sector pair in a configurable universe and ranks the tradeable ones.

Sector-grouped universe (screening.py:34-43) — only tests economically-related pairs, which gives a rationale and keeps the test count low.
Train/test split (screening.py:228-230) — every statistic is computed on a training window; the test window is held out to check whether cointegration persists out-of-sample.
Per-pair stats (test_pair): correlation pre-filter, Engle-Granger cointegration (tries both regression directions and keeps the stronger), ADF on the spread, and half-life of mean reversion via an OU/AR(1) fit (half_life).
Multiple-testing correction (screening.py:268-272) — exactly the Type I concern from your README. It computes a Bonferroni threshold and flags which pairs clear it.
Composite ranking by coint p-value + ADF p-value + half-life.


# On screening

Observation	Why it matters
KO/PEP didn't make the list	Confirms your original pair has no edge — your screen correctly rejects it.
6 pairs pass the in-sample filter (AMZN/GOOGL, MA/V, …)	Looks promising at first glance.
coint_p_bonferroni_ok is False for all of them	None survive multiple-testing correction — 91 tests at α=0.05 means ~4-5 false positives expected by chance.
coint_p_test is 0.17–0.97 for all	The cointegration does not hold out-of-sample. 
In-sample cointegration is largely a mirage here.

That last point is the whole lesson of quantitative pairs research, and your project now demonstrates it rather than ignoring it. The honest write-up — "I screened 91 pairs, found 6 in-sample hits, and showed none survived Bonferroni or held out-of-sample" — is far more impressive than a cherry-picked Sharpe.

So basically, I first tested these but they didn't work so I expanded the universe

Refer to METHODOLOGY.md

# Questions:
Why do we use log prices? What would happen if we used raw prices?

# Procedure:
1. Are data cointegrated?
2. If so, what is the hedge ratio? (Through regression)
3. Construct the spread