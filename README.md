# Data Science Mini Project

![alt text](https://github.com/kokoontech/data-science-mini-project-2020--h-ttfd/blob/master/housesbanner.png)

## Description
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

The dataset has 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, US.

Training and test datasets can be found in /data/.

## Skills
Creative feature engineering 

Advanced regression techniques like random forest and gradient boosting

## Goal
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

We want to minimise the Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

## Reporting and Approach
Rationale behind each decision must be explained. It isn't necessary to write algorithms from scratch, and feel free to leverage common libraries such as scikit-learn. Preferably submit in python, but other languages are acceptable. Data visualisation and exploration is encouraged. A well written and well thought out approach with modest performances is preferable over performance alone.

## Submission File Format and Coding Environment

The coding environment used and the experimental results should be well documented and easily reproducible.

The file should contain a header and have the following format:
```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
etc.
```
see /data/sample_submission.csv for an example of the correct submission format.

Upload your completed project to *your* GitHub, and then paste a link to the repository in the Coderbyte form along with any comments you have about your solution.

Good luck!
