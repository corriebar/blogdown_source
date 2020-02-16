# The Bayesian Appreciation of Wine and Tea
I came across the paper "The Analysis of Experimental Data: The Appreciation of Tea and Wine" by Dennis Lindley which goes over some problematic aspects of Fisher's famous tea cup experiment. The experiment is as follows:
Muriel Bristol, a friend of Fisher, claims that she can distinguish between tea where the milk has been added before the tea, and tea where the milk has been added after. Now everyone knows someone (or is this someone) that claims to be able to distinguish between Coca Cola and Pepsi, tea made with soft and with hard water (that's me) or cheap and expensive wine. So you've probably found yourself already in some experimental setup in which you tried to test someone's skill to discriminate. Setups on how to test vary, but one possible is to give pairs of cups of tea (we stick with the tea for now), one with the milk added before and the other with the milk added after. For each pair, note if the tested person could discriminate correctly or not, that is was she right $R$ or wrong $W$.
Fisher gave Lady Bristol 6 pairs of cups of tea and she got the following result: $RRRRRW$, that is, only the last pair was wrong. Does this mean she has discriminatory power or not?
Now Fisher proceeds as follows: He considers the null hypothesis, that is, the hypothesis that she cannot distinguish between the two and just guesses. Under that hypothesis, with probability $p=0.5$, she guesses correctly. In that case, the sequence $RRRRRW$ has probability $Pr(RRRRRW) = 0.5^6 = \frac{1}{64}$. But actually, every other sequence has the same probability. Instead, we're interested in the probability of guessing only one pair wrong, that is, all permutations of $RRRRRW$. There are 6 such permutations, hence $Pr(\text{ 1 pair wrong}) = 6 \cdot 0.5^6 = \frac{6}{64} = 0.094$. This still does not quite work. In the case of just guessing, the most probable outcome would be to have one half of pairs correct, the other half wrong. But this is not quite the outcome we expect to see most of the time, most of the time, we expect to see _around half_ of pairs correct. So instead of considering the probability of getting 5 of 6 pairs correct when just guessing, we now consider the probability of getting 5 or more out of 6 pairs correct. That is, the probability of all results as, or more, extreme than the result observed.
For 6 observations, this is the probability of getting only wrong or no wrong: $Pr(\text{ 1 wrong }) + Pr(\text{ no  wrong }) = \frac{7}{64} = 0.109$. Since someone in science at some point decided, that probabilities more than 5% are not enough to reject the null hypothesis, i.e. the resulsts are not significant. That's it Lady Bristol, you can't distinguish tea with milk from milk with tea.

But wait, says the paper, and this is where it gets interesting. Why was it only 6 cups? Did Lady Bristol need to go to a meeting? Or was the experiment setup to try as many pairs of cups until a mistake is made?
So what, you may say, the result is the same, isn't it?
Actually, it is not. In this case, the results more extreme than the result observed are different.
In terms of probability distributions, before we were dealing with a binomial distribution, now instead we're dealing with a negative binomial distribution: How many successes are observed before the first failure?
To compute the probability of observing 5 or more successes before the first failure, here just a quick reminder of some Calculus I facts:
$$\sum_{i=1}^{\infty} \( \frac{1}{} \)^i $$
is a geometric series and it converges. More generally:
$$\sum_{i=1}^{\infty} r^i = 1 - \frac{1}{1-r} $$
if $|r| <1$. Hence, the sum above with $r=\frac{1}{2}$ is 1. So to compute the probability of observing 5 or more successes before the first failure, we calculate the probability of having less succeses and substract from 1:
$$Pr(\text{ 5 ore more successes }) = \frac{1}{32} = 0.31$$
Voila, our result is significant! Lady Bristol can distinguish the two different ways of making tea.

Somehow, this is not very satisfactory. 
Let's try the Bayesian way instead.
