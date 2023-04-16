# CSCI 567 Machine Learning - Final Project

Team `CSCI567_id30`:

- Parsa Hejabi - [@ParsaHejabi](https://github.com/ParsaHejabi)
- Armin Abdollahi
- Mahmoudreza Dehghan

Competition Link: [DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/)

Leaderboard: [Link](https://www.drivendata.org/competitions/57/nepal-earthquake/leaderboard/)

---

## Project Description

We're trying to predict the ordinal variable `damage_grade`, which represents a level of damage to the building that was hit by the earthquake. There are 3 grades of the damage:

- `1` represents low damage
- `2` represents a medium amount of damage
- `3` represents almost complete destruction

## Features

The dataset mainly consists of information on the buildings' structure and their legal ownership. Each row in the dataset represents a specific building in the region that was hit by Gorkha earthquake.

There are 39 columns in this dataset, where the `building_id` column is a unique and random identifier. The remaining 38 features are described in the section below. Categorical variables have been obfuscated random lowercase ascii characters. The appearance of the same character in distinct columns does **not** imply the same original value.

| Name                                                 | Type        | Description                                                                                                                                                                              |
| ---------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id` | int         | geographic region in which building exists, from largest (level 1) to most specific sub-region (level 3). Possible values: `level 1`: `0-30`, `level 2`: `0-1427`, `level 3`: `0-12567`. |
| `count_floors_pre_eq`                                | int         | number of floors in the building before the earthquake.                                                                                                                                  |
| `age`                                                | int         | age of the building in years.                                                                                                                                                            |
| `area_percentage`                                    | int         | normalized area of the building footprint.                                                                                                                                               |
| `height_percentage`                                  | int         | normalized height of the building footprint.                                                                                                                                             |
| `land_surface_condition`                             | categorical | surface condition of the land where the building was built. Possible values: `n`, `o`, `t`.                                                                                              |
| `foundation_type`                                    | categorical | type of foundation used while building. Possible values: `h`, `i`, `r`, `u`, `w`.                                                                                                        |
| `roof_type`                                          | categorical | type of roof used while building. Possible values: `n`, `q`,`x`.                                                                                                                         |
| `ground_floor_type`                                  | categorical | type of the ground floor. Possible values: `f`, `m`, `v`, `x`, `z`.                                                                                                                      |
| `other_floor_type`                                   | categorical | type of constructions used in higher than the ground floors (except of roof). Possible values: `j`, `q`, `s`, `x`.                                                                       |
| `position`                                           | categorical | position of the building. Possible values: `j`, `o`, `s`, `t`.                                                                                                                           |
| `plan_configuration`                                 | categorical | building plan configuration. Possible values: `a`, `c`, `d`, `f`, `m`, `n`, `o`, `q`, `s`, `u`.                                                                                          |
| `has_superstructure_adobe_mud`                       | binary      | flag variable that indicates if the superstructure was made of Adobe/Mud.                                                                                                                |
| `has_superstructure_mud_mortar_stone`                | binary      | flag variable that indicates if the superstructure was made of Mud Mortar - Stone.                                                                                                       |
| `has_superstructure_stone_flag`                      | binary      | flag variable that indicates if the superstructure was made of Stone.                                                                                                                    |
| `has_superstructure_cement_mortar_stone`             | binary      | flag variable that indicates if the superstructure was made of Cement Mortar - Stone.                                                                                                    |
| `has_superstructure_mud_mortar_brick`                | binary      | flag variable that indicates if the superstructure was made of Mud Mortar - Brick.                                                                                                       |
| `has_superstructure_cement_mortar_brick`             | binary      | flag variable that indicates if the superstructure was made of Cement Mortar - Brick.                                                                                                    |
| `has_superstructure_timber`                          | binary      | flag variable that indicates if the superstructure was made of Timber.                                                                                                                   |
| `has_superstructure_bamboo`                          | binary      | flag variable that indicates if the superstructure was made of Bamboo.                                                                                                                   |
| `has_superstructure_rc_non_engineered`               | binary      | flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.                                                                                       |
| `has_superstructure_rc_engineered`                   | binary      | flag variable that indicates if the superstructure was made of engineered reinforced concrete.                                                                                           |
| `has_superstructure_other`                           | binary      | flag variable that indicates if the superstructure was made of any other material.                                                                                                       |
| `legal_ownership_status`                             | categorical | legal ownership status of the land where building was built. Possible values: `a`, `r`, `v`, `w`.                                                                                        |
| `count_families`                                     | int         | number of families that live in the building.                                                                                                                                            |
| `has_secondary_use`                                  | binary      | flag variable that indicates if the building was used for any secondary purpose.                                                                                                         |
| `has_secondary_use_agriculture`                      | binary      | flag variable that indicates if the building was used for agricultural purposes.                                                                                                         |
| `has_secondary_use_hotel`                            | binary      | flag variable that indicates if the building was used as a hotel.                                                                                                                        |
| `has_secondary_use_rental`                           | binary      | flag variable that indicates if the building was used for rental purposes.                                                                                                               |
| `has_secondary_use_institution`                      | binary      | flag variable that indicates if the building was used as a location of any institution.                                                                                                  |
| `has_secondary_use_school`                           | binary      | flag variable that indicates if the building was used as a school.                                                                                                                       |
| `has_secondary_use_industry`                         | binary      | flag variable that indicates if the building was used for industrial purposes.                                                                                                           |
| `has_secondary_use_health_post`                      | binary      | flag variable that indicates if the building was used as a health post.                                                                                                                  |
| `has_secondary_use_gov_office`                       | binary      | flag variable that indicates if the building was used fas a government office.                                                                                                           |
| `has_secondary_use_use_police`                       | binary      | flag variable that indicates if the building was used as a police station.                                                                                                               |
| `has_secondary_use_other`                            | binary      | flag variable that indicates if the building was secondarily used for other purposes.                                                                                                    |

## Performance Metric

We are predicting the level of damage from 1 to 3. The level of damage is an ordinal variable meaning that ordering is important. This can be viewed as a _classification_ or an _ordinal regression_ problem. (Ordinal regression is sometimes described as an problem somewhere in between classification and regression.)

To measure the performance of our algorithms, we'll use the **F1 score** which balances the precision and recall of a classifier. Traditionally, the F1 score is used to evaluate performance on a binary classifier, but since we have three possible labels we will use a variant called the **micro averaged F1 score**.

$$
F_{micro}=\frac{2 \cdot P_{micro} \cdot R_{micro}}{P_{micro}+R_{micro}}
$$

Where

$$
P_{micro}=\frac{\sum_{k=1}^{3} TP_{k}}{\sum_{k=1}^{3} (TP_{k}+FP_{k})}, \quad R_{micro}=\frac{\sum_{k=1}^{3} TP_{k}}{\sum_{k=1}^{3} (TP_{k}+FN_{k})}
$$

and $TP_{k}$, $FP_{k}$, and $FN_{k}$ are the number of true positives, false positives, and false negatives for class $k$ respectively. $k$ represents each class in $1, 2, 3$.

In Python, you can easily calculate this loss using `sklearn.metrics.f1_score` with the keyword argument `average='micro'`. Here are some references that discuss the micro-averaged F1 score further:

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Blog post](http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html)

## Submission Format

The format for the submission file is two columns with the `building_id` and the `damage_grade`. The data type of `damage_grade` is an integer with values of $1,2,3$
, **so make sure there is no decimal point and no other numbers in your submission**. For example `1` would be valid, and `1.0` would **not**.

For example, if you predicted:

| building_id | damage_grade |
| ----------- | ------------ |
| 11456       | 2            |
| 16528       | 3            |
| 3253        | 1            |
| 18614       | 2            |
| 15444       | 3            |

The first few lines of the `.csv` file that you submit would look like:

```csv
building_id,damage_grade
11456,2
16528,3
3253,1
18614,2
1544,3
```

## Getting Started

### Prerequisites

- [Python 3.11.3](https://www.python.org/downloads/release/python-3113/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Visual Studio Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Visual Studio Code Black Formatter Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [Visual Studio Code Pylint Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint)
- [Git](https://git-scm.com/downloads)

### Installation

1. Clone the repo

```sh
git clone git@github.com:ParsaHejabi/USC-CSCI567-MachineLearning.git
```

2. Check your Python version and make sure it is `3.11.3`

```sh
python3 --version
```

3. Create a virtual environment

```sh
python3 -m venv venv
```

4. Activate the virtual environment

```sh
source venv/bin/activate
```

5. Update pip

```sh
python3 -m pip install --upgrade pip
```

6. Install the requirements

```sh
pip install -r requirements.txt
```

7. Install the pre-commit hooks

```sh
pre-commit install
```
