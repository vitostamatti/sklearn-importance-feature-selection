# Scikit-Learn Importance Feature Selector

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/vitostamatti/sklearn-importance-feature-selection.svg)](https://github.com/vitostamatti/sklearn-importance-feature-selection/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/vitostamatti/sklearn-importance-feature-selection.svg)](https://github.com/vitostamatti/sklearn-importance-feature-selection/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

## 📝 Table of Contents

- [About](#about)
- [Setup](#setup)
- [Usage](#usage)



## About <a name = "about"></a>

To complete soon.

## Setup <a name = "setup"></a>

To get started, clone this repo and check that you have all requirements installed.

```
git clone https://github.com/vitostamatti/sklearn-importance-feature-selection.git
pip install .
``` 

## Usage <a name = "usage"></a>

```
from importance_feature_selector import ImportanceFeatureSelector

selector = ImportanceFeatureSelector()
X_selected = selector.fit_transform(X, y)

```

In the [notebooks](/notebooks/) directory you can find examples.


## Roadmap

- [X] First commit


## License
[MIT](LICENSE.txt)