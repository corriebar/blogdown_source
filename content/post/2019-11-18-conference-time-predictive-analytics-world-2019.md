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
---

In the last two years, I pretty much only went to very techie conferences, such as the PyData Berlin, the PyCon or the SatRday (this year I was even a speaker). They're all great conferences, organized by awesome people and I will definitely go again but this fall I decided to try out some new conference and check out the Predictive Analytics World in Berlin. First, because it's alsways good to try out new things and also because in the last months I became more aware how important a business focus is, which frankly isn't a focus in Python talks about how to deploy machine learning models. The Predictive Analytics World has a very clear focus on business, it's even in their logo. So much, that when I went to the conference venue this morning, that I thought I had dressed to casual, not wearing a business suit but luckily most of the serious dressed people where from a different conference in the same venue. I didn't see many people wearing t-shirts but also not many in full business attire. 

So here a short summary of my favorite talks:

## Introduction to Federated and Privacy Preserving Analytics & AI
I've already heard about Federated Learning (namely from this [fun comic](https://federated.withgoogle.com/)) but not too much so I really enjoyed this talk by Robin Röhm. The talk only gave a high-level overview about different methods but it gave good intuitions on how the various methods worked and also why this is important. I mean, not that we wouldn't already know as data scientists that data privacy is important but to be honest, it's not really a favorite topic and even though I personally think data privacy is really important, professionally all that data privacy stuff seems to get in the way of the cool analyses and models. So to me, federated learning is very exciting because it let's you have both: train a machine learning model and still preserve privacy. For example, I was recently working in a project where multiple companies all had similar data and it would have been so cool to put them all in one pot and train a model on them but then for legal reasons this was unthinkable. With federated learning, one could have a central model, send it to the different data pots, train locally and only send the weights back. There are different kinds of federated learning, for example instead of splitting up the data horizontally (each party has different rows but the same columns) one could also split up the data vertically where then each party has different features i.e. columns but for the same entities.
´
Some other important methods mentioned in the talk were [differential privacy](http://www.jetlaw.org/journal-archives/volume-21/volume-21-issue-1/differential-privacy-a-primer-for-a-non-technical-audience/), [homomorphic encryption](https://www.wired.com/2014/11/hacker-lexicon-homomorphic-encryption/) and [privacy preserving record linkage](https://www.data61.csiro.au/en/Our-Research/Our-Work/Privacy-Preserving-Record-Linkage). The last two don't seem to be super mature yet (or at least not with widespread tool support) but differential privacy is definitely an important concept to be aware of.

## Obtaining Uncertainty Estimates from Neural Networks Using Tensorflow Probability
As a Bayesian myself, I very much liked this talk by Sigrid Keydana. She first makes the case that we want more than just point estimates and also explains why the outputs from neural networks are not probabilities. One way to get around this is to use dropout uncertainty. There's a nice [blog post](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) by Yarin Gal about how dropout approximate uncertainty estimates. 

A more direct way to use uncertainty estimates is by using a Bayesian network and this is where we use Tensorflow Probability. As Sigrid works for RStudio, she demonstrated everything using R code (piping the model layers, very cool!). What I especially liked is, how she stressed the difference between aleatoric uncertainty and epistemic uncertainty. Aleatoric uncertainty is inherent to your problem, for example the variation due to the measurement process and it won't go away by collecting more data. Whereas epistemic uncertainty happens when you don't have enough data so it could potentially be fixed by collecting more data. I recommend you to check out her [blog posts](https://blogs.rstudio.com/tensorflow/posts/2019-11-13-variational-convnet/) on which the talk was partly based.

## Honoroable mentions
There was a kind of buzzwordy talk about how to become an AI-ready company. As I've been thinking a lot recently about how to get data science better embedded into an organization, this talk was very appreciated. They had a few cool ideas, some pretty standard (have an AI community), some very cute (a data science/AI adventcalendar) and the one I remember best was called _competetive news_: Having a company newsletter whith news about what companies in similar businesses have already implemented. It thus gets people thinking "hmm could we maybe do something similar" but also makes upper management listen. And as an added bonus, for these already-implemented-by-your-competitor use cases you know that they are doable. 