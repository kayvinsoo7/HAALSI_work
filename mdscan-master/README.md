# Multidimensional subset scanning (mdscan)

Given a dataset `D` with outcomes `Y` and discretized features `X`. Also given `E` to be a set of expectations or 'normal' values for `Y`, and `F` to be an expectation-based scoring statistic that measures the amount of anomalousness between subgroup observations and their expectations. MDScan efficiently identifies the most anomalous subgroup; `S^*`.

The scoring statistic maybe parametric (such as Bernoulli or Poisson) or non-parametric (such as Berkjones), and this use of a particular scoring statistics is based on the assumptions of the distribution of the outcome.

MDSS has been applied to identifying predictive bias at a group level where the null hypothesis is that the given prediction's odds are correct for all subgroups in the dataset and the alternative hypothesis assumes some constant multiplicative bias in the odds for some given subgroup. MDSS has also been applied to identify sub-populations who, as a group, have outcomes that significantly diverge from the overall population. 

### Usage

MDScan currently supports three scoring functions. These scoring functions usage are described below:
- *BerkJones*: Non-parametric scoring function. To be used for all of the four types of outcomes supported - binary, continuous, nominal, ordinal.
- *Bernoulli*: Parametric scoring function. To used for two of the four types of outcomes supported - binary and nominal.
- *Poisson*: Parametric scoring function. To be used for three of the four types of outcomes supported - binary, continuous, and ordinal.

Note, non-parametric scoring functions can only be used for datasets where the expectations are constant.

The type of outcomes must be provided using the mode keyword argument. The definition for the four types of outcomes supported are provided below:
- Binary: Yes/no outcomes. Outcomes must 0 or 1.
- Continuous: Continuous outcomes. Outcomes could be any real number.
- Nominal: Multiclass outcomes with no rank or order between them. Outcomes must be a finite set of integers with dimensionality <= 1,000.
- Ordinal: Multiclass outcomes that are ranked in a specific order. Outcomes must be positive integers.

You can find usecases that show how to use MDSS in the `examples` folder.

