---
title: 'Conference Time: Predictive Analytics World 2019'
author: Corrie
date: '2019-11-18'
slug: conference-time-predictive-analytics-world-2019
categories:
  - conference
tags:
  - conference
  - predictive analytics
comments: yes
image: images/tea_with_books.jpg
menu: ''
share: yes
aliases:
  - /post/conference-time-predictive-analytics-world-2019
---

In the last two years, I pretty much only went to very technical conferences, such as the PyData Berlin, the PyCon or the SatRday. They're all great conferences, organized by awesome people and I will definitely go again but this fall I decided to try out a new conference and check out the Predictive Analytics World in Berlin. First, because it's always good to try out new things and also because in the last months I was wondering a lot how data teams can be made more useful, somehow more aligned with the business challenges, which frankly isn't talked much in Python talks about how to deploy machine learning models. The Predictive Analytics World has a very clear focus on business, it's even in their logo. So much, that when I went to the conference venue the first day, I thought I had dressed to casual by not wearing a business suit but luckily most of the seriously dressed people where from a different conference in the same venue. 
Anyway, I was hoping to find some hints and tips at this conference on how to have a more effective data team.

So here a short summary of my favorite talks:

## Introduction to Federated and Privacy Preserving Analytics & AI
I've already heard about Federated Learning (namely from this [fun comic](https://federated.withgoogle.com/)) but not too much so I really enjoyed this talk by Robin Röhm. The talk only gave a high-level overview about different methods but it gave good intuitions on how the various methods worked and also why this is important. I mean, not that we wouldn't already know as data scientists that data privacy is important but to be honest, it's not really a favorite topic and even though I personally think data privacy is really important, professionally all that data privacy stuff seems to get in the way of the cool analyses and models. So to me, federated learning is very exciting because it let's you have both: train a machine learning model and still preserve privacy. For example, I was recently working in a project with multiple companies involved. They all had similar data and it would have been so cool to put all the data in one pot and train a model on them but then for legal and privacy reasons, this was unthinkable. With federated learning, one could have a central model, send it to the different data pots, train locally and only send the weights back. There are different kinds of federated learning, for example instead of splitting up the data horizontally (each party has different rows but the same columns) one could also split up the data vertically where then each party has different features i.e. columns but for the same entities.

Some other important methods mentioned in the talk were [differential privacy](http://www.jetlaw.org/journal-archives/volume-21/volume-21-issue-1/differential-privacy-a-primer-for-a-non-technical-audience/), [homomorphic encryption](https://www.wired.com/2014/11/hacker-lexicon-homomorphic-encryption/) and [privacy preserving record linkage](https://www.data61.csiro.au/en/Our-Research/Our-Work/Privacy-Preserving-Record-Linkage). The last two don't seem to be super mature yet (or at least not with widespread tool support) but differential privacy is definitely an important concept to be aware of.

## Obtaining Uncertainty Estimates from Neural Networks Using Tensorflow Probability
Being a Bayesian aficionado myself, I very much liked this talk by Sigrid Keydana. She first makes the case that we want more than just point estimates and also explains why the outputs from neural networks are not probabilities. One way to get around this is to use dropout uncertainty. There's a nice [blog post](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) by Yarin Gal about how dropout approximates uncertainty estimates. 

A more direct way to get uncertainty estimates is by using a Bayesian network and this is where we use Tensorflow Probability. As Sigrid works for RStudio, she demonstrated everything using R code (piping the model layers, very cool!). What I especially liked is, how she stressed the difference between aleatoric uncertainty and epistemic uncertainty. Aleatoric uncertainty is inherent to your problem, for example the variation due to the measurement process and it won't go away by collecting more data. Whereas epistemic uncertainty happens when you don't have enough data so it can potentially be fixed by collecting more data. I recommend you to check out her [blog posts](https://blogs.rstudio.com/tensorflow/posts/2019-11-13-variational-convnet/) on which the talk was partly based.

## Honorable mentions
- The slightly buzzwordy talk about how to become an AI-ready company. As I've been thinking a lot recently about how to get a data team better embedded into an organization, I was looking forward to this talk. They had a few cool ideas, some pretty standard (have an AI community), some very cute (an AI advent calendar) and the one I remember best was called _competitive news_: Having a company newsletter with news about what companies in similar businesses have already implemented. It thus gets people thinking "hmm could we maybe do something similar" but also makes upper management listen. And as an added bonus, your competitor already proved that this use cases is doable. You can check out the speaker's articles on [medium](https://medium.com/@TheJuliaButter).
- One talk about forecasting demand mentioned how they use [Stan](https://mc-stan.org/) and hierarchical models to plan the fresh food stock. Unfortunately the talk was very high-level and only talking about some general problems in forecasting demand (to be fair, the talk was only around 20min). I would have liked to hear more about the model used. In general (also from other talks), my impression is that some problems specific to forecasting make it quite suitable for Bayesian methods.
- Forecasting demand in a bakery using weather data: apparently people buy less cake when it rains. Odd, rain sounds like the perfect tea-and-cake-on-the-couch weather to me.
- The closing keynote: Encouraging the crowd, after two days of talks about predictive analytics and AI, to use less complex models and instead use more heuristics and intuition. It was kind of like a good joke but there's definitely truth to it, I had some projects myself where neural nets and the like barely outperformed the heuristic baseline and we ended up implementing the baseline.

## Summary
Quite many talks were either directly about how to embed a data team in a company and how to actually get value out of data science or they mentioned it somehow. I think it's a big problem that many companies face: how do you actually get some value out of your data and your data scientists. Just hiring some data scientists (whatever that means) and throwing data at them is clearly not enough. No one seemed to have a definite solution for these kind of problems, most were still in a phase of trying out various things, sharing tips on what worked well so far and what less so. For these talks, it was worth to go for me but I didn't hear too many things I didn't already know. In the end, my favorite talks were pretty technical and would've fit just fine in e.g. a PyData conference. Many talks had lots of AI power point slides, that is, a bit buzzwordy and hype, with not too many new ideas or concepts.
So all in all, it was nice but not sure I'd go again.