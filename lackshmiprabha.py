{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "lackshmiprabha.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "32cbYQ4hkZuB"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import sklearn"
      ],
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "47ArO1Aol7Sd"
      },
      "source": [
        "from sklearn.datasets import load_boston\n",
        "df = load_boston()"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1Br8eU2imuX6",
        "outputId": "2699c196-6701-45f3-95dc-65b769866a38"
      },
      "source": [
        "df.keys()  # Returns all the keys of the dataset dictionary"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MM_H4ctpnpwB",
        "outputId": "5d371b27-d09b-48a7-a844-a3dc7165c600"
      },
      "source": [
        "print(df.DESCR)  # Info about the dataset"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            ".. _boston_dataset:\n",
            "\n",
            "Boston house prices dataset\n",
            "---------------------------\n",
            "\n",
            "**Data Set Characteristics:**  \n",
            "\n",
            "    :Number of Instances: 506 \n",
            "\n",
            "    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n",
            "\n",
            "    :Attribute Information (in order):\n",
            "        - CRIM     per capita crime rate by town\n",
            "        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n",
            "        - INDUS    proportion of non-retail business acres per town\n",
            "        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n",
            "        - NOX      nitric oxides concentration (parts per 10 million)\n",
            "        - RM       average number of rooms per dwelling\n",
            "        - AGE      proportion of owner-occupied units built prior to 1940\n",
            "        - DIS      weighted distances to five Boston employment centres\n",
            "        - RAD      index of accessibility to radial highways\n",
            "        - TAX      full-value property-tax rate per $10,000\n",
            "        - PTRATIO  pupil-teacher ratio by town\n",
            "        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n",
            "        - LSTAT    % lower status of the population\n",
            "        - MEDV     Median value of owner-occupied homes in $1000's\n",
            "\n",
            "    :Missing Attribute Values: None\n",
            "\n",
            "    :Creator: Harrison, D. and Rubinfeld, D.L.\n",
            "\n",
            "This is a copy of UCI ML housing dataset.\n",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n",
            "\n",
            "\n",
            "This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n",
            "\n",
            "The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\n",
            "prices and the demand for clean air', J. Environ. Economics & Management,\n",
            "vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n",
            "...', Wiley, 1980.   N.B. Various transformations are used in the table on\n",
            "pages 244-261 of the latter.\n",
            "\n",
            "The Boston house-price data has been used in many machine learning papers that address regression\n",
            "problems.   \n",
            "     \n",
            ".. topic:: References\n",
            "\n",
            "   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n",
            "   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "2d6KBRDUroGL",
        "outputId": "36ca0dd2-b29e-4926-d150-d8e8da27a42c"
      },
      "source": [
        "boston = pd.DataFrame(df.data, columns=df.feature_names)\n",
        "boston.head()"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>CRIM</th>\n",
              "      <th>ZN</th>\n",
              "      <th>INDUS</th>\n",
              "      <th>CHAS</th>\n",
              "      <th>NOX</th>\n",
              "      <th>RM</th>\n",
              "      <th>AGE</th>\n",
              "      <th>DIS</th>\n",
              "      <th>RAD</th>\n",
              "      <th>TAX</th>\n",
              "      <th>PTRATIO</th>\n",
              "      <th>B</th>\n",
              "      <th>LSTAT</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.00632</td>\n",
              "      <td>18.0</td>\n",
              "      <td>2.31</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.538</td>\n",
              "      <td>6.575</td>\n",
              "      <td>65.2</td>\n",
              "      <td>4.0900</td>\n",
              "      <td>1.0</td>\n",
              "      <td>296.0</td>\n",
              "      <td>15.3</td>\n",
              "      <td>396.90</td>\n",
              "      <td>4.98</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.02731</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>6.421</td>\n",
              "      <td>78.9</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>396.90</td>\n",
              "      <td>9.14</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.02729</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>7.185</td>\n",
              "      <td>61.1</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>392.83</td>\n",
              "      <td>4.03</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.03237</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>6.998</td>\n",
              "      <td>45.8</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>394.63</td>\n",
              "      <td>2.94</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.06905</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>7.147</td>\n",
              "      <td>54.2</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>396.90</td>\n",
              "      <td>5.33</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT\n",
              "0  0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98\n",
              "1  0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14\n",
              "2  0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03\n",
              "3  0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94\n",
              "4  0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33\n",
              "\n",
              "[5 rows x 13 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "D_4JM97Fsbjq",
        "outputId": "edc29bbd-73eb-4319-880f-be14f4eceb06"
      },
      "source": [
        "boston['MEDV'] = df.target\n",
        "boston.head()"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>CRIM</th>\n",
              "      <th>ZN</th>\n",
              "      <th>INDUS</th>\n",
              "      <th>CHAS</th>\n",
              "      <th>NOX</th>\n",
              "      <th>RM</th>\n",
              "      <th>AGE</th>\n",
              "      <th>DIS</th>\n",
              "      <th>RAD</th>\n",
              "      <th>TAX</th>\n",
              "      <th>PTRATIO</th>\n",
              "      <th>B</th>\n",
              "      <th>LSTAT</th>\n",
              "      <th>MEDV</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.00632</td>\n",
              "      <td>18.0</td>\n",
              "      <td>2.31</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.538</td>\n",
              "      <td>6.575</td>\n",
              "      <td>65.2</td>\n",
              "      <td>4.0900</td>\n",
              "      <td>1.0</td>\n",
              "      <td>296.0</td>\n",
              "      <td>15.3</td>\n",
              "      <td>396.90</td>\n",
              "      <td>4.98</td>\n",
              "      <td>24.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.02731</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>6.421</td>\n",
              "      <td>78.9</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>396.90</td>\n",
              "      <td>9.14</td>\n",
              "      <td>21.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.02729</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7.07</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.469</td>\n",
              "      <td>7.185</td>\n",
              "      <td>61.1</td>\n",
              "      <td>4.9671</td>\n",
              "      <td>2.0</td>\n",
              "      <td>242.0</td>\n",
              "      <td>17.8</td>\n",
              "      <td>392.83</td>\n",
              "      <td>4.03</td>\n",
              "      <td>34.7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.03237</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>6.998</td>\n",
              "      <td>45.8</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>394.63</td>\n",
              "      <td>2.94</td>\n",
              "      <td>33.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.06905</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.18</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.458</td>\n",
              "      <td>7.147</td>\n",
              "      <td>54.2</td>\n",
              "      <td>6.0622</td>\n",
              "      <td>3.0</td>\n",
              "      <td>222.0</td>\n",
              "      <td>18.7</td>\n",
              "      <td>396.90</td>\n",
              "      <td>5.33</td>\n",
              "      <td>36.2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  MEDV\n",
              "0  0.00632  18.0   2.31   0.0  0.538  ...  296.0     15.3  396.90   4.98  24.0\n",
              "1  0.02731   0.0   7.07   0.0  0.469  ...  242.0     17.8  396.90   9.14  21.6\n",
              "2  0.02729   0.0   7.07   0.0  0.469  ...  242.0     17.8  392.83   4.03  34.7\n",
              "3  0.03237   0.0   2.18   0.0  0.458  ...  222.0     18.7  394.63   2.94  33.4\n",
              "4  0.06905   0.0   2.18   0.0  0.458  ...  222.0     18.7  396.90   5.33  36.2\n",
              "\n",
              "[5 rows x 14 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 407
        },
        "id": "Evj0KBvftr0B",
        "outputId": "4608fe4e-f48f-43ca-d923-7fac946ace84"
      },
      "source": [
        "boston.isnull()"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>CRIM</th>\n",
              "      <th>ZN</th>\n",
              "      <th>INDUS</th>\n",
              "      <th>CHAS</th>\n",
              "      <th>NOX</th>\n",
              "      <th>RM</th>\n",
              "      <th>AGE</th>\n",
              "      <th>DIS</th>\n",
              "      <th>RAD</th>\n",
              "      <th>TAX</th>\n",
              "      <th>PTRATIO</th>\n",
              "      <th>B</th>\n",
              "      <th>LSTAT</th>\n",
              "      <th>MEDV</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>501</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>502</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>503</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>504</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>505</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>506 rows × 14 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "      CRIM     ZN  INDUS   CHAS    NOX  ...    TAX  PTRATIO      B  LSTAT   MEDV\n",
              "0    False  False  False  False  False  ...  False    False  False  False  False\n",
              "1    False  False  False  False  False  ...  False    False  False  False  False\n",
              "2    False  False  False  False  False  ...  False    False  False  False  False\n",
              "3    False  False  False  False  False  ...  False    False  False  False  False\n",
              "4    False  False  False  False  False  ...  False    False  False  False  False\n",
              "..     ...    ...    ...    ...    ...  ...    ...      ...    ...    ...    ...\n",
              "501  False  False  False  False  False  ...  False    False  False  False  False\n",
              "502  False  False  False  False  False  ...  False    False  False  False  False\n",
              "503  False  False  False  False  False  ...  False    False  False  False  False\n",
              "504  False  False  False  False  False  ...  False    False  False  False  False\n",
              "505  False  False  False  False  False  ...  False    False  False  False  False\n",
              "\n",
              "[506 rows x 14 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D576x8xnt5Mk",
        "outputId": "983cb4a9-0a72-424a-f224-c33891cdf8ca"
      },
      "source": [
        "boston.isnull().sum()"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "CRIM       0\n",
              "ZN         0\n",
              "INDUS      0\n",
              "CHAS       0\n",
              "NOX        0\n",
              "RM         0\n",
              "AGE        0\n",
              "DIS        0\n",
              "RAD        0\n",
              "TAX        0\n",
              "PTRATIO    0\n",
              "B          0\n",
              "LSTAT      0\n",
              "MEDV       0\n",
              "dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1uvDhVg1uk_j",
        "outputId": "d75d0c87-bf4f-4c60-e152-9c5074cd0a36"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X = boston.drop('MEDV',axis=1)\n",
        "Y = boston['MEDV']\n",
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state=5)\n",
        "print(X_train.shape)\n",
        "print(X_test.shape)\n",
        "print(Y_train.shape)\n",
        "print(Y_test.shape)"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(430, 13)\n",
            "(76, 13)\n",
            "(430,)\n",
            "(76,)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pmKlJhMyxWfn"
      },
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error"
      ],
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5-dv9Wz7y0oR",
        "outputId": "c1ed84f6-a8f9-41fb-fae9-adc4a7dd318d"
      },
      "source": [
        "## FITTING MODEL ON THE TRAINING DATASET\n",
        "\n",
        "lin_model = LinearRegression()\n",
        "\n",
        "lin_model.fit(X_train, Y_train)"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6WZFk1SpzriP",
        "outputId": "7cdd1187-c7de-4566-e48c-b6e84ccfacd1"
      },
      "source": [
        "Y_train_predict = lin_model.predict(X_train)\n",
        "rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_predict)))\n",
        "\n",
        "print(\"The model perfomance for training set\")\n",
        "print('RMSE is {}'.format(rmse))\n",
        "print(\"\\n\")\n",
        "\n",
        "# on testing set\n",
        "y_test_predict = lin_model.predict(X_test)\n",
        "rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))\n",
        "\n",
        "print(\"The model perfomance for testing set\")\n",
        "print('RMSE is {}'.format(rmse))"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "The model perfomance for training set\n",
            "RMSE is 4.710901797319796\n",
            "\n",
            "\n",
            "The model perfomance for testing set\n",
            "RMSE is 4.687543527902972\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}