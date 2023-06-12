# 2022 League of Legends Esports Match Data Analysis: Prediction modeling


By Ricky Zhu(r2zhu@ucsd.edu) and Tony Guo(xig003@ucsd.edu)

---

# Framing the Problem

## Prediction Problem

We are interested for the following prediction:
**Predict which role(Top-lane, Jungle, Support, etc) a player played given their post-game data in League of Legends.**

Prediction Type: **Multiclass Classification**

## Response Variable

The player's **position**: Top Laner(top), Jungle(jng), Mid Laner(mid), Bot Laner(bot), Support for Bot Laner(sup)

We choose position as our response variable because we are interested in looking any possible correlation between a player's performance data and their position in the team. 

## Metric

We choose **Accuracy Score** as our metric. Accuracy Score is a simple and intuitive metric that measures the overall correctness of predictions. It calculates the ratio of correctly classified samples to the total number of samples. While Accuracy Score is sensitive to class imbalances, our five positions in League of Legends are balanced and have the same number of samples, where the each game for every team has equally distributed positions to every player, so we don't need to worry about it.

### Justification about Time of Prediction

We are using post-game data to predict the position of player, while the position of player is determined at the start of the game. As we do the predictions after the games are finished, we can then use all the data in the dataset.


## Data Cleaning

**Rows**: A League of Legends eSports player's match statistics in a single game in 2022. 

**Number of rows**: 149400

**Relevant Columns:**

1. **damagetakenperminute**: Damage taken per minute, indicating the amount of damage a player have taken from attacks of others in one minute.

2. **gamelength**: How long did a player played in a single eSport match, marked in seconds.

3. **position**: The player's position in a single eSport match. League of Legends only has **five positions: Top Laner(top), Jungle(jng), Mid Laner(mid), Bot Laner(bot), Support for Bot Laner(sup).** Note: In a League of Legends match, there are 3 lanes total and jungle areas. Top laner goes to top lane, Mid laners goes to mid lane, and Jungle clears Jungle area and potentially help other lanes. Bot Laners and Support goes bot lane. 

4. **kills**: The number of kills a player have in a single eSport match.

5. **assists**: The number of assists a player have in a single eSport match.

6. **death**: The number of deaths a player have in a single eSport match.

7. **earned gpm**: Earned gold per minute by player in a single game. **In League of Legends, gold is used to buy items in order to make your league character stronger.**

8. **cspm**: Creep score per minute, indicating the number of minions killed by a player per minute.

9. **wpm**: Ward per minute, indicating the amount of wards a player placed in one minute.

10. **vspm**: Vision score per minute, a indicator of a player's contribution to their team's control of the map by either placing wards or removing enemy wards.

11. **dpm**: Damage per minute, indicating the amount of damage a player made in one minute.

12. **result**: The result of a single game in one match, either win(Denoted by 1) or lose(Denoted by 0). 

13. **monsterkills**: The number of monsters killed in one game, including dragons/baron. 

---

# Baseline Model

## Model Description

* The pipeline consists of **two main steps**: **data preprocessing** and **classification**.
* The data **preprocessing** step is performed by a **ColumnTransformer**, which applies specific transformations to different subsets of columns in the dataset.
* The preprocessing step includes standard scaling of the "wpm" feature using StandardScaler and one-hot encoding of the "result" feature using OneHotEncoder.
* The remainder parameter is set to **'passthrough'**, indicating that any remaining columns not explicitly transformed will be passed through without any changes.
* The classification step utilizes a **DecisionTreeClassifier**, which is responsible for making predictions based on the preprocessed data.

## Features in the Model

* The features used for training and prediction are all columns in the cleaned dataframe except for the "position" column.
* We focus on the following two features: 
  * **Quantitative** Feature: "wpm"
  * **Nominal** Feature: "result" (after one-hot encoding)
  
## Encoding

* The "wpm" feature is **standardized** using **StandardScaler**, which scales the values to have zero mean and unit variance.
* The "result" feature is one-hot encoded using **OneHotEncoder**, creating **binary columns** for each unique category of the "result" feature.

## Performance

For 10 times, we **randomly** split whole dataset into a train/test split of **7:3** and fit the pipeline model. Then we calculate the **accuracy scores** on the test set. 
*We calculate the score for ten times because the result may vary and we want to see if the accuracy scores are consistent.*

After experiments, we notice that the accuracy scores are around **0.72**. Thus we can conclude that the model's performance is **not good enough**, which indicates that we can't accurately predict player positions solely with the information about ward per minute and result of the game.

Also, there may be **other limitations with the DecisionTreeClassifier** we are using, since decision trees have a tendency to **overfit the training data**. 
* *Overfitting occurs when the model captures noise and irrelevant patterns in the training data, leading to poor generalization on unseen data. This is more likely to happen with multiclass classification because the decision boundaries can become more intricate, and the model might memorize the training examples instead of learning meaningful patterns.*

---

# Final Model

## Features:

1. **Standardized** damage per minute, damage taken per minute, ward per minute, and vision score per minute
  * Standard Scaler brings the data to a common scale as different features or columns in a dataset may have different scales. This difference in scales can cause certain features to dominate the learning process. StandardScaler equalizes the scales by bringing all features to a similar range, ensuring that no single feature dominates the others.

2. **Robust scaled** kill-death-assist ratio, earned gold per minute, creep score per minute, and total monster kills
  * RobustScaler scales the data by removing the median and scaling it according to the interquartile range, making it **robust to outliers** while still **preserves the distribution shape of the original data.** This can be particularly useful when the data contains skewed distributions or contains outliers that are meaningful to the problem.
  * We use RobustScaler on these columns because they have huge outliers. 
  
3. **Binarized** game length
  * Binarizer is used to transform a continuous variable into a binary representation based on a specific cutoff point. As the 'gamelength' column is continuous, it can be useful.
  * We believe that using binarized feature of short and long game(> 30 min, or 1800 sec) will help us better predict the position of a player is playing because short games usually will produce skewed results(Ex: Support doing more damage than bot laners like AD carry), while long game can produce a better result in predicting as everyone’s damage during the game can be more stable and useful in predicting player’s position post-game.
  
4. **One-Hot-Encoded** game result
  * Although the 'result' column has a int64 datatype and consists of 1s and 0s, it is better to see it as a categorical column of winning and losing.
  * This transformer preserves the categorical nature of the data and prevents the model from assuming any ordinal relationship between categories.
  * OneHotEncoder allows for easy interpretation of the impact of each category on the model's predictions.

## Modeling Algorithm

- We use **RandomForestClassifier**. RandomForestClassifier combines multiple decision trees to form an ensemble. Each tree is constructed using a different random subset of the training data and random feature subsets. At each node of the decision tree, a random subset of features is considered for splitting.

- We perform a **GridSearchCV** because it allows us to define a grid of **hyperparameter** values to search over. It exhaustively tries all possible combinations of these values and evaluates the model's performance using cross-validation. This helps in finding the best combination of hyperparameters that optimize the model's performance on the given data.

- **criterion: gini.** The criterion determines the quality of a split in each decision tree within the random forest. The value 'gini' indicates that the Gini impurity measure is used to evaluate the splits. Gini impurity measures the degree of impurity or uncertainty in a node based on the class distribution.

- **max_depth: 15.** The max_depth parameter specifies the maximum depth allowed for each decision tree in the random forest. It limits the number of levels in the tree from the root node to the leaf nodes. In this case, the maximum depth is set to 15, meaning that each tree will have a maximum of 15 levels.

- **min_samples_split: 10.** The min_samples_split parameter determines the minimum number of samples required to split an internal node. If the number of samples at a node is less than this value, the node will not be split further and will become a leaf node. Here, the minimum number of samples required for splitting a node is set to 10.

## Performance

After using additional features on top of those used in our Baseline Model and changing the classifier from DecisionTreeClassifier to RandomForestClassifier, throughout many runs, the final model accuracy scores have a **steady increase of approximately 0.07**(i.e., from 0.72 to 0.79). 

In comparison with DecisionTreeClassifier, RandomForestClassifier helps in **reducing overfitting** and increasing the diversity of the trees in the ensemble. Random forests are also **less sensitive to noisy data and outliers**. What's more, RandomForestClassifier provides a feature importance measure, which helps identify the most important features for prediction. Therefore, RandomForestClassifier improves our model performance in terms of accuracy.

---

# Fairness Analysis

In order to test the fairness of our Final Model, we try to answer the question “does my model perform worse for games of **X** length than it does for games of **Y** length?”, for an interesting choice of X and Y.

Choice of X & Y: **long & short**

Evaluation Meric: **accuracy score**

We do it with **permutation** testings.

**Null Hypothesis**: Our model is fair. The classifier's accuracy is the same for both long games(longer than 30 minutes, 
or to say 1800 seconds) and short games, and any differences are due to chance.
**Alternative Hypothesis**: Our model is unfair. The classifier's accuracy is higher for short games.

Choice of **Test Statistic**: Difference between accuracy score of Long and Short games(Short - Long).

**Significane Level**: 0.05

<iframe src="assets/fairness2.html" width=800 height=600 frameBorder=0></iframe>

**P-value**: Our p-value vary from 0.9 to almost 1.0 in 100 permutation testings, which are all greater than 0.05.

**Conclusion**: we **fail to reject the null hypothesis**, which suggests that our model **may be fair**, and the classifier's accuracy may be the same for both long games and short games. This fail of rejection is based on our statistical analysis, specifically the calculation of the p-value, which is a measure of the likelihood of obtaining a result as extreme as, or more extreme than, the one observed if the null hypothesis is true. In our case, the obtained p-value is bigger than our commonly used significance level of 0.05 (p-value > 0.05).

However, it is important to consider the **limitations** of our study, such as the specific context, population size, and potential confounding variables, when interpreting these results. Further research and analysis may be warranted to gain a deeper understanding of whether we can predict a player's position based on their post-game data.

--- 


