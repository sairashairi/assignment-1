{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "04e62b87-d719-4f56-8569-29a97cfe1794",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "saira shairi.....\n"
     ]
    }
   ],
   "source": [
    "print (\"saira shairi.....\")       \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "182f30dc-53df-4c93-8075-2e5501e75a7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[27, 125, 343, 512, 1728]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "#Task 1.1:\n",
    "nums = [3, 5, 7, 8, 12]\n",
    "\n",
    "\n",
    "cubes = []\n",
    "\n",
    "\n",
    "for num in nums:\n",
    "    cubes.append(num ** 3)\n",
    "\n",
    "\n",
    "print(cubes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "678b2ff7-01ea-48c8-96fd-2794ffb1e822",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'parrot': 2, 'goat': 4, 'spider': 8, 'crab': 10}\n"
     ]
    }
   ],
   "source": [
    "#Task 1.2:\n",
    "my_dict = {}\n",
    "\n",
    "\n",
    "my_dict['parrot'] = 2\n",
    "my_dict['goat'] = 4\n",
    "my_dict['spider'] = 8\n",
    "my_dict['crab'] = 10\n",
    "\n",
    "\n",
    "print(my_dict)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "c1878b68-e6f2-4a99-98b7-50416c4e20c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "parrot: 2 legs\n",
      "goat: 4 legs\n",
      "spider: 8 legs\n",
      "crab: 10 legs\n",
      "Total number of legs: 24\n"
     ]
    }
   ],
   "source": [
    "#Task 1.3:\n",
    "my_dict = {\n",
    "    'parrot': 2,\n",
    "    'goat': 4,\n",
    "    'spider': 8,\n",
    "    'crab': 10\n",
    "}\n",
    "\n",
    "# Initialization\n",
    "total_legs = 0\n",
    "\n",
    "# Looping\n",
    "for animal, legs in my_dict.items():\n",
    "\n",
    "    print(f'{animal}: {legs} legs')\n",
    "    \n",
    "    \n",
    "    total_legs += legs\n",
    "\n",
    "# Printing\n",
    "print(f'Total number of legs: {total_legs}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "5059dd41-9716-4e54-99e5-91a2ef518f11",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3, 9, 4, [8, 6])\n"
     ]
    }
   ],
   "source": [
    "#Task 1.4:\n",
    "# Creating\n",
    "A = (3, 9, 4, [5, 6])\n",
    "\n",
    "# Modifying\n",
    "A[3][0] = 8\n",
    "\n",
    "# Printing\n",
    "print(A)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "41ffe70e-ff88-4df8-9709-aaf8ac16917c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Task 1.5: Delete the tuple \n",
    "del A\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "cccf34cf-d102-491c-a541-d1a6e912e1e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of occurrences of 'p': 2\n"
     ]
    }
   ],
   "source": [
    "#Task 1.6:\n",
    "# Creation \n",
    "B = ('a', 'p', 'p', 'l', 'e')\n",
    "\n",
    "# Counting number of 'p'\n",
    "count_p = B.count('p')\n",
    "\n",
    "# Printing\n",
    "print(f\"Number of occurrences of 'p': {count_p}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "39351e9c-4cf7-413f-9fb0-7a609790a9f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index of 'l': 3\n"
     ]
    }
   ],
   "source": [
    "#Task 1.7:\n",
    "# Creatin\n",
    "B = ('a', 'p', 'p', 'l', 'e')\n",
    "\n",
    "# Finding index 'l'\n",
    "index_l = B.index('l')\n",
    "\n",
    "# Printing\n",
    "print(f\"Index of 'l': {index_l}\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "6bb42281-ec6d-4529-8f6b-f39bd1bd61ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Matrix A:\n",
      " [[ 1  2  3  4]\n",
      " [ 5  6  7  8]\n",
      " [ 9 10 11 12]]\n"
     ]
    }
   ],
   "source": [
    " # Task 2\n",
    "import numpy as np\n",
    "\n",
    "# 2.1 Convert matrix A into numpy array\n",
    "A = np.array([[1, 2, 3, 4],\n",
    "              [5, 6, 7, 8],\n",
    "              [9, 10, 11, 12]])\n",
    "print(\"Matrix A:\\n\", A)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "8595b57e-2cc9-4a09-af4c-f7d118d85443",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Subarray b:\n",
      " [[2 3]\n",
      " [6 7]]\n",
      "Shape of b: (2, 2)\n"
     ]
    }
   ],
   "source": [
    "# 2.2 Use slicing \n",
    "b = A[:2, 1:3]  # First 2 rows, columns 1 and 2\n",
    "print(\"Subarray b:\\n\", b)\n",
    "print(\"Shape of b:\", b.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "55e66c51-b9c0-426a-8653-32939148116b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty matrix C:\n",
      " [[ 1  2  3  4]\n",
      " [ 5  6  7  8]\n",
      " [ 9 10 11 12]]\n"
     ]
    }
   ],
   "source": [
    "# 2.3 \n",
    "C = np.empty_like(A)\n",
    "print(\"Empty matrix C:\\n\", C)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "bd21b331-34bc-454b-898c-abf308ad8155",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Matrix C after adding z to each column:\n",
      " [[ 2  3  4  5]\n",
      " [ 5  6  7  8]\n",
      " [10 11 12 13]]\n"
     ]
    }
   ],
   "source": [
    "# 2.4 \n",
    "z = np.array([1, 0, 1])\n",
    "for i in range(A.shape[1]):  # Iterate over columns\n",
    "    C[:, i] = A[:, i] + z\n",
    "print(\"Matrix C after adding z to each column:\\n\", C)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "0688c106-a562-42c3-a722-3d3d2e184541",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sum of matrices X and Y:\n",
      " [[ 6  8]\n",
      " [10 12]]\n"
     ]
    }
   ],
   "source": [
    "# 2.5\n",
    "X = np.array([[1, 2], [3, 4]])\n",
    "Y = np.array([[5, 6], [7, 8]])\n",
    "v = np.array([9, 10])\n",
    "sum_XY = np.add(X, Y)\n",
    "print(\"Sum of matrices X and Y:\\n\", sum_XY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "1b031f81-1a10-40d6-a8c3-4907488d7d0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Element-wise multiplication of matrices X and Y:\n",
      " [[ 5 12]\n",
      " [21 32]]\n"
     ]
    }
   ],
   "source": [
    "# 2.6 \n",
    "product_XY = np.multiply(X, Y)\n",
    "print(\"Element-wise multiplication of matrices X and Y:\\n\", product_XY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "681d139b-b437-42a5-9965-7921c474b06d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Element-wise square root of matrix Y:\n",
      " [[2.23606798 2.44948974]\n",
      " [2.64575131 2.82842712]]\n"
     ]
    }
   ],
   "source": [
    "# 2.7 \n",
    "sqrt_Y = np.sqrt(Y)\n",
    "print(\"Element-wise square root of matrix Y:\\n\", sqrt_Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "f30f7da9-2564-4ae4-836b-0d8b31a57d8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dot product of matrix X and vector v:\n",
      " [29 67]\n"
     ]
    }
   ],
   "source": [
    "# 2.8 \n",
    "dot_Xv = np.dot(X, v)\n",
    "print(\"Dot product of matrix X and vector v:\\n\", dot_Xv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "d9113c57-9410-464e-99d1-7dbd0a835018",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sum of each column of matrix X:\n",
      " [4 6]\n"
     ]
    }
   ],
   "source": [
    "# 2.9 \n",
    "sum_columns_X = np.sum(X, axis=0)\n",
    "print(\"Sum of each column of matrix X:\\n\", sum_columns_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "e8db862a-e8a3-4c2b-9cd8-77d69738f230",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Velocity: 5.0 meters per second\n"
     ]
    }
   ],
   "source": [
    "#Task 3:\n",
    "#Task 3.1:\n",
    "def Compute(distance, time):\n",
    "    \"\"\"Calculating velocity given distance and time.\"\"\"\n",
    "    if time == 0:\n",
    "        raise ValueError(\"Time cannot be zero to avoid division by zero.\")\n",
    "    velocity = distance / time\n",
    "    return velocity\n",
    "\n",
    "\n",
    "distance = 100  \n",
    "time = 20       \n",
    "velocity = Compute(distance, time)\n",
    "print(f\"Velocity: {velocity} meters per second\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "9e6bdf01-084e-45c0-8f74-49a36aa38efa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Product of even numbers: 46080\n"
     ]
    }
   ],
   "source": [
    "#Task 3.2:\n",
    "def mult(numbers):\n",
    "    \"\"\"Calculating product of all numbers in the list.\"\"\"\n",
    "    product = 1\n",
    "    for num in numbers:\n",
    "        product *= num\n",
    "    return product\n",
    "\n",
    "# Creation of list \n",
    "even_num = [2, 4, 6, 8, 10, 12]\n",
    "\n",
    "\n",
    "product_of_evens = mult(even_num)\n",
    "print(f\"Product of even numbers: {product_of_evens}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "bafaeab8-fdcd-4970-bdd6-18ae4abb5c82",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataFrame:\n",
      "    C1  C2  C3  C4\n",
      "0   1   6   7   7\n",
      "1   2   7   9   5\n",
      "2   3   5   8   2\n",
      "3   5   4   6   8\n",
      "4   5   8   5   8\n"
     ]
    }
   ],
   "source": [
    "#Task 4:\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "# Create the DataFrame\n",
    "data = {\n",
    "    'C1': [1, 2, 3, 5, 5],\n",
    "    'C2': [6, 7, 5, 4, 8],\n",
    "    'C3': [7, 9, 8, 6, 5],\n",
    "    'C4': [7, 5, 2, 8, 8]\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "print(\"DataFrame:\\n\", df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "30bbd6cc-8913-49d6-868d-e65be23f0d27",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "First two rows of the DataFrame:\n",
      "    C1  C2  C3  C4\n",
      "0   1   6   7   7\n",
      "1   2   7   9   5\n"
     ]
    }
   ],
   "source": [
    "#Task 4.1:\n",
    "# Printing\n",
    "print(\"\\nFirst two rows of the DataFrame:\\n\", df.head(2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "65d893da-d14e-4af4-9f99-bab255d983a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Second column:\n",
      " 0    6\n",
      "1    7\n",
      "2    5\n",
      "3    4\n",
      "4    8\n",
      "Name: C2, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#Task 4.2:\n",
    "# Print 2nd \n",
    "print(\"\\nSecond column:\\n\", df['C2'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "2533c622-da9c-4039-99d5-815476d523e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "DataFrame after renaming column 'C3' to 'B3':\n",
      "    C1  C2  B3  C4\n",
      "0   1   6   7   7\n",
      "1   2   7   9   5\n",
      "2   3   5   8   2\n",
      "3   5   4   6   8\n",
      "4   5   8   5   8\n"
     ]
    }
   ],
   "source": [
    "#Task 4.3:\n",
    "# Renaming\n",
    "df.rename(columns={'C3': 'B3'}, inplace=True)\n",
    "print(\"\\nDataFrame after renaming column 'C3' to 'B3':\\n\", df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "9ed18a42-77d4-4825-b885-17275dda0691",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "DataFrame after adding column 'Sum':\n",
      "    C1  C2  B3  C4  Sum\n",
      "0   1   6   7   7    0\n",
      "1   2   7   9   5    0\n",
      "2   3   5   8   2    0\n",
      "3   5   4   6   8    0\n",
      "4   5   8   5   8    0\n"
     ]
    }
   ],
   "source": [
    "#Task 4.4:\n",
    "df['Sum'] = 0\n",
    "print(\"\\nDataFrame after adding column 'Sum':\\n\", df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "60b8d130-6a8c-499e-88e5-954f22862785",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "DataFrame with 'Sum' column updated:\n",
      "    C1  C2  B3  C4  Sum\n",
      "0   1   6   7   7   21\n",
      "1   2   7   9   5   23\n",
      "2   3   5   8   2   18\n",
      "3   5   4   6   8   23\n",
      "4   5   8   5   8   26\n"
     ]
    }
   ],
   "source": [
    "#Task 4.5:\n",
    "df['Sum'] = df.sum(axis=1)\n",
    "print(\"\\nDataFrame with 'Sum' column updated:\\n\", df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "178f5f20-a4fb-4fa2-baff-eddb87a25427",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "d0a11805-1009-40f6-bae7-f99c48cf21e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\Shairi\\\\Desktop\\\\pyth1'"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "a0050faa-376a-4f38-ba23-1263f6e67181",
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
       "      <th>Gender</th>\n",
       "      <th>F_Color</th>\n",
       "      <th>B_Month</th>\n",
       "      <th>Weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>Black</td>\n",
       "      <td>March</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>Red</td>\n",
       "      <td>March</td>\n",
       "      <td>72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Male</td>\n",
       "      <td>Black</td>\n",
       "      <td>October</td>\n",
       "      <td>53</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Female</td>\n",
       "      <td>Blue</td>\n",
       "      <td>February</td>\n",
       "      <td>63</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>black</td>\n",
       "      <td>October</td>\n",
       "      <td>80</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gender F_Color   B_Month  Weight\n",
       "0    Male   Black     March      77\n",
       "1    Male     Red     March      72\n",
       "2    Male   Black   October      53\n",
       "3  Female    Blue  February      63\n",
       "4    Male   black  October       80"
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "variable= pd.read_csv(r\"C:\\Users\\Shairi\\Desktop\\pyth1\\hello_sample.csv\")\n",
    "variable.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "f3f38cb6-8c6c-48a5-a052-651c5519157e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#task 4.6\n",
    "import pandas as pd\n",
    "\n",
    "# Read the CSV file\n",
    "df = pd.read_csv('hello_sample.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "1c75761b-7186-4c01-9896-aaf73ade1f34",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Gender  F_Color     B_Month  Weight\n",
      "0     Male    Black       March      77\n",
      "1     Male      Red       March      72\n",
      "2     Male    Black     October      53\n",
      "3   Female     Blue    February      63\n",
      "4     Male    black    October       80\n",
      "5   Female    Black    december      97\n",
      "6     Male   Orange      August      60\n",
      "7   Female      Red   September      59\n",
      "8     Male     Blue        July      60\n",
      "9     Male     Blue         May      58\n",
      "10  Female      Red    October       60\n",
      "11    Male    Black   September      65\n",
      "12    Male    Green    December      85\n",
      "13    Male    Black       April      72\n",
      "14    Male    Green    January       90\n",
      "15    Male   Yellow   November       90\n",
      "16    Male    Black      August      54\n",
      "17    Male    Black     January      82\n",
      "18    Male  Mustard   December       60\n",
      "19  Female    Black    October       76\n",
      "20    Male    Black   February       67\n",
      "21    Male   Yellow       April      74\n",
      "22    Male      Red  September       66\n"
     ]
    }
   ],
   "source": [
    "#task 4.7\n",
    "print(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "id": "d1880511-5704-43c9-ade2-660ec16ca6e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Gender F_Color     B_Month  Weight\n",
      "21   Male  Yellow       April      74\n",
      "22   Male     Red  September       66\n"
     ]
    }
   ],
   "source": [
    "#task 4.8\n",
    "print(df.tail(2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "id": "7febf66e-641d-4c4f-a739-e7fdabb9cf8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 23 entries, 0 to 22\n",
      "Data columns (total 4 columns):\n",
      " #   Column   Non-Null Count  Dtype \n",
      "---  ------   --------------  ----- \n",
      " 0   Gender   23 non-null     object\n",
      " 1   F_Color  23 non-null     object\n",
      " 2   B_Month  23 non-null     object\n",
      " 3   Weight   23 non-null     int64 \n",
      "dtypes: int64(1), object(3)\n",
      "memory usage: 868.0+ bytes\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "#task 4.9\n",
    "print(df.info())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "edd7f1dc-6556-49ab-8f51-e72236774983",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(23, 4)\n"
     ]
    }
   ],
   "source": [
    "#task 4.10\n",
    "print(df.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "d31b1b24-cfd2-4c41-8f79-cb5f1db793be",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Gender  F_Color     B_Month  Weight\n",
      "2     Male    Black     October      53\n",
      "16    Male    Black      August      54\n",
      "9     Male     Blue         May      58\n",
      "7   Female      Red   September      59\n",
      "18    Male  Mustard   December       60\n",
      "10  Female      Red    October       60\n",
      "6     Male   Orange      August      60\n",
      "8     Male     Blue        July      60\n",
      "3   Female     Blue    February      63\n",
      "11    Male    Black   September      65\n",
      "22    Male      Red  September       66\n",
      "20    Male    Black   February       67\n",
      "13    Male    Black       April      72\n",
      "1     Male      Red       March      72\n",
      "21    Male   Yellow       April      74\n",
      "19  Female    Black    October       76\n",
      "0     Male    Black       March      77\n",
      "4     Male    black    October       80\n",
      "17    Male    Black     January      82\n",
      "12    Male    Green    December      85\n",
      "14    Male    Green    January       90\n",
      "15    Male   Yellow   November       90\n",
      "5   Female    Black    december      97\n"
     ]
    }
   ],
   "source": [
    "#task 4.11\n",
    "df_sorted = df.sort_values(by='Weight')\n",
    "print(df_sorted)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "3bc1c86d-0b63-4fa3-9038-ae59a59a5397",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gender     0\n",
      "F_Color    0\n",
      "B_Month    0\n",
      "Weight     0\n",
      "dtype: int64\n",
      "    Gender  F_Color     B_Month  Weight\n",
      "0     Male    Black       March      77\n",
      "1     Male      Red       March      72\n",
      "2     Male    Black     October      53\n",
      "3   Female     Blue    February      63\n",
      "4     Male    black    October       80\n",
      "5   Female    Black    december      97\n",
      "6     Male   Orange      August      60\n",
      "7   Female      Red   September      59\n",
      "8     Male     Blue        July      60\n",
      "9     Male     Blue         May      58\n",
      "10  Female      Red    October       60\n",
      "11    Male    Black   September      65\n",
      "12    Male    Green    December      85\n",
      "13    Male    Black       April      72\n",
      "14    Male    Green    January       90\n",
      "15    Male   Yellow   November       90\n",
      "16    Male    Black      August      54\n",
      "17    Male    Black     January      82\n",
      "18    Male  Mustard   December       60\n",
      "19  Female    Black    October       76\n",
      "20    Male    Black   February       67\n",
      "21    Male   Yellow       April      74\n",
      "22    Male      Red  September       66\n"
     ]
    }
   ],
   "source": [
    "#task 4.12\n",
    "# this check missing values\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# and now droping any rows with missing values\n",
    "df_cleaned = df.dropna()\n",
    "print(df_cleaned)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76c8a73b-5ff0-4e3f-bfd0-c02802c8827e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
