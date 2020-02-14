
# House-Cleaning: Getting rid of outlier with Gauss

Working with real-world data presents many challenges that sanitized data from text book simply don't have. One of the biggest is how to handle outlier. Now outlier can mean different things. Some outliers are very obvious: someone accidently pressed the zero-key a bit too long and now the standard two-room appartment in an average location has an offering price of three billion euro instead of the more reasonable 300,000€. If you know that the most expensive home in the world (the Buckingham Palace) is valued at around 1.3 billion euro then it is obvious that three billion euro cannot have been the correct offering price. Other cases can be more tricky. What is for example with the old villa in small town offered for 1€? Did the person forget to add some zeros? Or is it one of these run-down houses, maybe in a dying town that someone just wants to get rid of because one would need to invest half a million euro to make the house habitable again?
Similarly, the flat offered for rent for 15,000 euro per months is much more expensive than the average rent and thus might potentially be an outlier, but looking closer, it also has 300sqm and is in a really ncie area, so it might be just a very expensive flat.

So very roughly we can say that there are two categories of outliers:
- outliers that are not real, but instead result from an error or mistake
- outliers that are real, but look extremely different compared with other data.

Now, let's talk about how we can identify these outliers and how we can clean them from our data.

I recently scraped some rental offers from Immoscout24 and except for throwing away some duplicates that arose probably due to the way I scraped them, this is still very much the raw data and thus still has a bunch of outliers.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklego.mixture import GMMClassifier, GMMOutlierDetector

plt.style.use('corrie')
```


```python
d = pd.read_csv("/home/corrie/Documents/Projects/immoscout/data/immo_data.csv")
d["totalRent"] = np.where(d["totalRent"].isnull(), d["baseRent"], d["totalRent"])
```

The easiest way to see if there are any outliers in your data is to just plot it. Let's look at the total rent versus the living space:


```python
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(d["livingSpace"],d["totalRent"], s=2)
ax.set_xlabel("Living Space (in sqm)")
ax.set_ylabel("Total Rent (in €)")
ax.set_title("Rent vs Living Space")
plt.show()
```

Does this plot remind you of all the times you wanted to have a look at your data, see if there's any interesting pattern or some insight to be gained just to then be reminded of the fact that your data is still dirty? It definitely does for me. There are some flats or houses that are larger than 10,000 sqm which, according to [the Measure of Things](https://www.bluebulbprojects.com/MeasureOfThings/results.php?comp=area&unit=m2&amt=10000&sort=pr&p=1) is twice as big as Bill Gate's home. And that place with a rent of 10 million? Doesn't look right.

One way to get a nicer plot, is to just eyeball the axes, use whatever knowledge you have about rents and flats and pick some threshold above or below which you discard the data:


```python
too_large = d["livingSpace"] > 5000
too_expensive = d["totalRent"] > 2e5
alrightish = ~too_large & ~too_expensive
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(d["livingSpace"][alrightish],d["totalRent"][alrightish], s=2, alpha=0.8)
ax.set_xlabel("Living Space (in sqm)")
ax.set_ylabel("Total Rent (in €)")
ax.set_title("Rent vs Living Space\nNot too large nor too expensive")
plt.show()
```

It looks better than before but still not great. One could iteratively try smaller thresholds until the blob in the bottom left covers most of the plot area. I think this can be a valid way to obtain reasonable thresholds but might not be very feasible if you have more variables and it's also not quite satisfying if you like to automate stuff.
In this blog post, I want to describe two simple methods to prune outliers.

The first one is the interquartile range rule and might be familiar to you if you had some classes in statistics.

## Interquartile Range Rule
[Quartiles](https://en.wikipedia.org/wiki/Quartile), are important summary statistics of any continuous data. There are three quartiles, $Q_1$, $Q_2$ and $Q_3$, which are the 25th, 50th (the median) and 75th percentiles. The interquartile range $IQR$ is then the difference between the first and third quartile: 
$IQR = Q_3 - Q_1$
The interquartile range thus covers half of the data which is also why they're used for boxplots: the boxy part is exactly the interquartile range. 

A good rule of thumb is to say that every point above $Q_3 +1.5 QR$ or below $Q_1 - 1.5 IQR$ is an outlier.
In python, we can compute this as follows:


```python
def iqr(data):
    """compute the interquartile range (excluding nan)"""
    return np.nanquantile(data, 0.75) - np.nanquantile(data, 0.25)

def iqr_rule(data, factor=1.5):
    """returns an outlier filter mask using the iqr rule"""
    iqr_ = iqr(data)
    upper_fence = np.nanquantile(data, 0.75) + factor*iqr_
    lower_fence = np.nanquantile(data, 0.25) - factor*iqr_
    return (data <= upper_fence) & (data >= lower_fence)
```

For our example, the same plot as above after applying the IQR rule to both our variables then looks like this:


```python
alrightish = iqr_rule(d["livingSpace"]) & iqr_rule(d["totalRent"])
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(d["livingSpace"][alrightish],d["totalRent"][alrightish], s=6, alpha=0.6)
ax.set_xlabel("Living Space (in sqm)")
ax.set_ylabel("Total Rent (in €)")
ax.set_title("Rent vs Living Space\nIQR rule")
plt.show()
```

Nice! The blob now covers the whole plot area and we get a much better view of the largest chunk of the data.
However, there is a hard, rectangular edge. It looks like we cut off the data too generously and a considereable part of the data is now hidden from view because they're either larger than 140sqm or more expensive than 1750€ per month. And while I personally wouldn't rent a flat this expensive, it still sounds like very realistic data and actually not too unachievable. If I would have a flat share with three or four friends, 2000€ would be a reasonable rent in Berlin.

To adjust this rule of thumb, we can increase the factor with which we multiply the IQR. Instead of 1.5, we can use e.g. 2.5:


```python
iqr_rule_25 = lambda x: iqr_rule(x, factor=2.5)
alrightish = iqr_rule_25(d["livingSpace"]) & iqr_rule_25(d["totalRent"])
sns.jointplot(x=d["livingSpace"][alrightish],y=d["totalRent"][alrightish],
              s=6, alpha=0.6, height=9, marginal_kws=dict(bins=100) )
plt.show()
```

The plot has less cut-off than before but there's still a rectangular border.
I also added the histograms for each margin to get a look at the individual distributions. One can see that the distribution for the rent does not follow a normal distribution. The distribution for the living space also seems to be slightly right-skewed.
Furthermore, we know that both rents and living space should have values above zero. (There are actually a few observations that have a value of zero for the living space or the total rent. This is clearly not realistic.) So it is probably more reasonable to model both variables assuming a log-normal distribution. 
I manually assigned values of 0.5 to the observations that have a value of zero. Seems somehow reasonable to say that there's not much difference between the rent being 0€ or 50ct.


```python
d["livingSpace_m"] =  np.where(d["livingSpace"] <= 0, 0.5, d["livingSpace"])
d["totalRent_m"] = np.where(d["totalRent"] <= 0, 0.5, d["totalRent"])
logbins=np.logspace(0,np.log(10e3),500)
g = sns.jointplot(x=d["livingSpace_m"],y=d["totalRent_m"], 
                  s=6, alpha=0.6, height=9, marginal_kws=dict(bins=logbins)
                 )
g.ax_joint.set_yscale("log")
g.ax_marg_x.set_xscale("log")
g.ax_marg_y.set_xscale("log")
g.ax_joint.set_xscale("log")
g.ax_joint.set(xlabel="livingSpace", ylabel="totalRent")
plt.show()
```

Using the IQR rule on the transformed data, we get the following result:


```python
d["logRent"] = np.log(d["totalRent_m"])
d["logSpace"] = np.log(d["livingSpace_m"])

fig, ax  = plt.subplots(figsize=(20,9), nrows=1, ncols=2)
alrightish = iqr_rule_25(d["logSpace"]) & iqr_rule_25(d["logRent"])
d["outlier"] = np.where(alrightish, "no_outlier", "outlier")
max_space = d[d.outlier == "no_outlier"].livingSpace.max()
max_rent = d[d.outlier == "no_outlier"].totalRent.max()
sns.scatterplot(x="livingSpace_m", y="totalRent_m", hue="outlier", palette="Set1",
                hue_order = ["outlier", "no_outlier"], s=6,
                data=d,  alpha=0.6, ax=ax[1], linewidth=0)
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set(xlabel="livingSpace", ylabel="totalRent")
ax[1].set_title("Data with Outliers\n(on log-scale)")

ax[0].scatter(d["livingSpace"][alrightish],d["totalRent"][alrightish], s=6, alpha=0.6)
ax[0].set(xlabel="livingSpace", ylabel="totalRent")
ax[0].set_title("Data without Outliers\n(normal scale)")

plt.show()

```

On the left, we see the data after the so identified outliers. Compared to before, the rectangular borders don't seem as bad as before but they're still there. On the left, there's a hard cut throwing away places that are too small. The smallest place that is not considered an outlier has a size of 16.97sqm. Small indeed but it is hard to argue that there couldn't be smaller flats. One likely problem here is that the data also contains shared flats. The size in the offer would then be the size of the room but the rent would be relatively high compared with a flat of that size (which might not have enough space for a proper kitchen which lowers rent) because in a flat share one also pays for common areas.

On the right, I plotted all data and highlighted outliers in red. It shows that we not only lose data in the lower left corner but there's also a hard corner in the upper right. One could increase the factor used in the IQR rule but this would still not lead to optimal results.
One problem with the IQR rule is that we apply it separately on each variable. That is, the cut-off window will always be rectangular. In this example though, the two variables are highly correlated and thus a rectangular window is not a good fit. Better would be to use an oval cut-off window.
An oval window as for example obtained by a multivariate Gaussian distribution.

## Gaussian Mixtures for Outlier Detection


```python
from sklego.mixture import BayesianGMMOutlierDetector

exam = np.random.multivariate_normal([-10, 2], [[1.8, 1.7], [1.7, 2.3]], (1000,))
outlier = [[-7, -1], [-8, 7.5]]
exam = np.vstack([exam, outlier])
mod = BayesianGMMOutlierDetector(n_components=1, threshold=3.5, method="stddev").fit(exam)

df_ex= pd.DataFrame({"x1": exam[:, 0], "x2": exam[:, 1],
                   "loglik": mod.score_samples(exam), 
                   "prediction": mod.predict(exam).astype(str)})

fig, axes = plt.subplots(figsize=(18,7), nrows=1, ncols=2)
axes[0].scatter(df_ex.x1, df_ex.x2, c=df_ex.loglik, cmap="viridis_r", s=6)
axes[0].set_title("Multivariate Normal Distribution")


factor = 1.5
x1_max = np.nanquantile(exam[:,0], 0.75) + factor*iqr(exam[:,0])
x1_min = np.nanquantile(exam[:,0], 0.25) - factor*iqr(exam[:,0])

x2_max = np.nanquantile(exam[:,1], 0.75) + factor*iqr(exam[:,1])
x2_min = np.nanquantile(exam[:,1], 0.25) - factor*iqr(exam[:,1])


axes[1].scatter(df_ex.x1[df_ex.prediction == "-1"], df_ex.x2[df_ex.prediction == "-1"], c="red", label="low likelihood", s=6)
axes[1].scatter(df_ex.x1[df_ex.prediction == "1"], df_ex.x2[df_ex.prediction == "1"], c="steelblue", s=6)
axes[1].axvline(x=x1_max, c="grey", label="IQR rule", alpha=0.3)
axes[1].axvline(x=x1_min, c="grey", alpha=0.3)
axes[1].axhline(y=x2_max, c="grey", alpha=0.3)
axes[1].axhline(y=x2_min, c="grey", alpha=0.3)
axes[1].set_title("IQR rule vs Multivariate Outlier Detector")
axes[1].legend()
plt.show()
```

On the left are points sampled from a multivariate normal with a high correlation between the $x$ and $y$ variable, similarly as in our data. The points are colored by their log likelihood. I added one outlier by hand at $(-7, -1)$ and as you can see it has a much lower log likelihood compared to the other points. We can use this and classify all points with a very low likelihood as outliers.
On the right, we see the same points where now points with a low likelihood are in red. The grey lines gives the rectangular threshold as obtained from the IQR rule. There are quite a few points at the lower left and upper right corner that would be classified as outlier by the IQR rule whereas it would miss the outlier point I added manually.

Luckily, there's a package for that: [scikit-lego](https://github.com/koaning/scikit-lego). The package follows the scikit-learn API and adds some additional classifies (such as the Gaussian mixture classifier and outlier detector) but also useful transformers and a pipeline debugger.
The function I'm going to use, fits a multivariate Gaussian to our data, computes the likelihood for each point and points with a low likelihood are flagged as outlier. Vincent, one of the developer of scikit-lego, explains this in a few more sentences in his [talk](http://koaning.io/theme/notebooks/gaussian-progress.pdf) which I can recommend.


```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklego.mixture import BayesianGMMOutlierDetector

num_cols = ["logRent", "logSpace"]

p = 1
sample_size = int(len(d[num_cols]) * p)
sample_idx = np.random.choice(len(d[num_cols]), size=sample_size, replace = False)
```


```python
pipe = make_pipeline(StandardScaler(),
                     BayesianGMMOutlierDetector(n_components=1, threshold=1.5, method="stddev") )

pipe.fit(d[["logRent", "logSpace"]])
outlier = pipe.predict(d[["logRent", "logSpace"]])
```


```python
outlier = pipe.predict(d[num_cols])
d["outlier"] = np.where(outlier == -1, "outlier", "no_outlier")
```


```python
d.outlier.value_counts()
```


```python
fig, ax  = plt.subplots(figsize=(9,9))
ax = sns.scatterplot(x="livingSpace", y="totalRent", 
                data=d[d.outlier == "no_outlier"], alpha=0.6, ax=ax, linewidth=0, s=6)
plt.show()
```


```python
fig, ax  = plt.subplots(figsize=(9,9))
max_space = d[d.outlier == "no_outlier"].livingSpace.max()
max_rent = d[d.outlier == "no_outlier"].totalRent.max()
ax = sns.scatterplot(x="livingSpace", y="totalRent", hue="outlier", palette="Set1",
                hue_order = ["outlier", "no_outlier"], s=8,
                data=d,  alpha=0.6, ax=ax, linewidth=0)
ax.set_xlim(-40, max_space + 200)
ax.set_ylim(-1000, max_rent + 800)
plt.show()
```


```python

```


```python
pd.options.display.max_colwidth = 100
d[["serviceCharge", "baseRent", "totalRent", "livingSpace", "description"]][d.outlier == "outlier"].head(10)
```


```python
def prob_outlier(outlier_detector_pipe, data, iterations=30, p=0.01):
    sample_size = int(len(data) * p)

    outlier_ar = np.empty((0, len(data)) )
    for i in range(iterations):
        outlier_detector_pipe.fit(data.sample(sample_size))

        outlier_ar = np.append(outlier_ar,  [outlier_detector_pipe.predict(data)], axis=0)

    outlier = (outlier_ar == -1).mean(axis=0)
    return outlier
```


```python
num_cols = ["logRent", "logSpace"]

d["outlier"] = prob_outlier(pipe, d[num_cols])
```


```python
d.outlier.hist(log=True)

plt.show()
```


```python
np.sum(d["outlier"] >= 0.95)
```


```python
fig, ax  = plt.subplots(figsize=(9,9))
d["outlier"] = np.where(d["outlier"] >= 0.95, "outlier", "no_outlier")
max_space = d[d.outlier == "no_outlier"].livingSpace.max()
max_rent = d[d.outlier == "no_outlier"].totalRent.max()
sns.scatterplot(x="livingSpace", y="totalRent", hue="outlier", palette="Set1",
                hue_order = ["outlier", "no_outlier"], s=8,
                data=d,  alpha=0.5, ax=ax, linewidth=0)
ax.set_xlim(-40, max_space + 200)
ax.set_ylim(-1000, max_rent + 800)
plt.show()
```


```python
num_cols = ["logRent", ""]
```


```python
d["age"] = 2020.5 - d["yearConstructed"]
d["logAge"] = np.log(d["age"])
```


```python
d[d.age <= 0].yearConstructed.value_counts()
```


```python
mask = d["yearConstructed"].notnull()
mask = d["logAge"].notnull()
```


```python
d["outlier"] = np.nan
d["outlier"][mask] = prob_outlier(pipe, d[["logRent", "logAge", "logSpace"]][mask])
```


```python
d[d.outlier >= 0.95][["totalRent", "yearConstructed", "livingSpace", "description", "regio2"]]
```


```python
fig, ax  = plt.subplots(figsize=(9,9))
d["outlier"] = np.where(d["outlier"] >= 0.9, "outlier", "no_outlier")
max_space = d[(d.outlier == "no_outlier") & mask].livingSpace.max()
max_rent = d[(d.outlier == "no_outlier") & mask].totalRent.max()
sns.scatterplot(x="livingSpace", y="totalRent", hue="outlier", palette="Set1",
                hue_order = ["outlier", "no_outlier"],
                data=d[mask],  alpha=0.5, ax=ax, linewidth=0)
ax.set_xlim(-40, max_space + 200)
ax.set_ylim(-1000, max_rent + 800)
plt.show()
```

## Summary

- Outlier detection is hard: what constitutes an outlier if often not very obvious
- What to keep and what to throw away often depends on the use case
- Transforming highly skewed variables leads to better outcomes, in particular less false positives
- Outliers are rare by definition so fitting on a subset can lead to better results