# Thinking, fast and slow and Bayes
I'm currently reading the book "Thinking, Fast and Slow" by Daniel Kahneman. There were quite a few parts in the book that almost made me give up reading it (the whole first part about social priming is in particular bad when you know about the replication crisis in social psychology) but I also found parts that I enjoyed reading. The second part of the book deals with how humans are not very good with statistics. 
Let us consider the following scenario:
> A cab was involved in a hit-and-run accident at night. Tow cab companies, the Green and the Blue, operate in the city. You are given the following data:
> - 85% of the cabs in the city are Green and 15% are Blue.
> - A witness identified the cab as Blue. The court tested the reliability of the witness under the circumstances that existed on the night of the accident and concluded that the witness correctly identified each one of the two colors 80% of the time and failed 20% of the time.

> What is the probability that the cab involved in the accident was Blue rather than Green?

Now a very common answer to this question is somewhere around 80%. If your last statistic lecture on the Bayesian Theorem is still very fresh, maybe you recognized that this answer is wrong.
If we would write this problem in formulas, it would like this:
$$ P(\text{Cab of the accident} = green) = 0.85 $$ and
$$ P(\text{Cab of the accident} = blue) = 0.15. $$
The information given by the witness would be encoded as
$$ P(\text{Witness says, Cab } = blue \, | \text{ Cab} = blue) = 0.80 $$
$$ P(\text{Witness says, Cab } = blue \, | \text{ Cab} = green ) = 0.20. $$
That is, the probability that the witness identifies the cab as blue  when the cab was blue is 80%.
Now we can plug these values in the Bayesian Theorem to obtain the probability that the cab was blue, given the information of the witness.
Recall:
$$ P(A| B) = \frac{p(B|A) P(A)}{P(B)} ,$$
where for our case
$$ A = \text{ Cab of the accident is blue } $$
and
$$ B = \text{ Witness says, that the cab of the accident was blue } .$$
To compute $P(B)$, we use the fact that $P(B) = P(B|A)P(A) + P(B| \text{ not } A)P(\text{ not } A)$.
We thus get:
$$ P(B) = 0.8 \cdot 0.15 + 0.2 \cdot 0.85 = 0.29 .$$
Pluggin in all these values to Bayes Formula, we get:
$$ P(A | B ) = \frac{0.8 \cdot 0.15}{0.29} = 0.4138 .$$
So most people give intuitively a very wrong answer. The problem is, that people underestimate the base rate (in this case, the base rate is the occurence of green and blue cabs in the city). Most often, the base rate is completely disregarded and one only considers the evidence (here, the testimonial of the witness). But it is important to always relate your evidence with the base rate.
An intersting twist happens when you change the scenario given in the beginning just slightly.
The story is the same, only the data is presented differently:
> You are given the following data:
> - The two companies operate the same number of cabs, but Green cabs are involved in 85% of accidents.
> - The information about the witness is as in the previous version.

The probabilities in the formula representation don't change a single bit and Bayes formula gives the same result, but the intuitive answer people give does change. Now, most people actually give an estimate close to the accurate solution. Kahneman argues, that people find it easier to use the base rate information when it is not just statistical information but forms a story. In the second case, the story people would form, is such that green cabs dirve much more recklessly and thus the base rate given in the second scenario provides us a causal story line.

I have been studying Bayesian Statistics and inference quite a bit in the last months, so when I read the story, I was already familiar with the statistical set-up and how it works. So while I didnt't fall into this "trap", it is good to be aware that most people would intuitively give very different probability estimates. Since my job as a Data Scientist is to also communicate my results and help other people understand them, I find it valuable insight and was thinking how can use a story line to help make unintutive results easier to understand. This is a bit tricky, since a good story



TODO: 
- Bayes theorem in terms of hypothesis and evidence