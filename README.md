
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
