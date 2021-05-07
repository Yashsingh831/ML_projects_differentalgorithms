{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "salary_data=pd.read_csv('salary.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(32561, 5)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salary_data.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
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
       "      <th>age</th>\n",
       "      <th>education.num</th>\n",
       "      <th>capital.gain</th>\n",
       "      <th>hours.per.week</th>\n",
       "      <th>income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>90</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>82</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>66</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>54</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>41</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  education.num  capital.gain  hours.per.week income\n",
       "0   90              9             0              40  <=50K\n",
       "1   82              9             0              18  <=50K\n",
       "2   66             10             0              40  <=50K\n",
       "3   54              4             0              40  <=50K\n",
       "4   41             10             0              40  <=50K"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salary_data.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "income_set=set(salary_data['income'])\n",
    "salary_data['income']=salary_data['income'].map({'<=50K':0,'>50K':1}).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
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
       "      <th>age</th>\n",
       "      <th>education.num</th>\n",
       "      <th>capital.gain</th>\n",
       "      <th>hours.per.week</th>\n",
       "      <th>income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>90</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>82</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>66</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>54</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>41</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  education.num  capital.gain  hours.per.week  income\n",
       "0   90              9             0              40       0\n",
       "1   82              9             0              18       0\n",
       "2   66             10             0              40       0\n",
       "3   54              4             0              40       0\n",
       "4   41             10             0              40       0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salary_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=salary_data.iloc[:,:-1]\n",
    "Y=salary_data.iloc[:,-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler=StandardScaler()\n",
    "X_train=scaler.fit_transform(X_train)\n",
    "X_test=scaler.transform(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "error=[]\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "for i in range (1,40):\n",
    "   model=KNeighborsClassifier(n_neighbors=i)\n",
    "   model.fit(X_train,y_train)\n",
    "   y_predict=model.predict(X_test)\n",
    "   error.append(np.mean(y_predict!=y_test))\n",
    "plt.figure(figsize=(12,6))\n",
    "plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)\n",
    "plt.title('error rate k value')\n",
    "plt.xlabel('k value')\n",
    "plt.ylabel('mean error')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=16)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "model=KNeighborsClassifier(n_neighbors=16,metric='minkowski',p=2)\n",
    "model.fit(X_train,y_train)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "above 50k\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "y_pred=model.predict(scaler.transform([[40,50,7,9]]))\n",
    "if y_pred==1:\n",
    "    print('above 50k')\n",
    "else:\n",
    "    print('below 50k')\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_test=model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5871,  322],\n",
       "       [1183,  765]], dtype=int64)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(y_test,y_pred_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8151332760103182"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "accuracy_score(y_test,y_pred_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
