# NearestNeighbor-without-scikit-learn
In python I wrote the Nearest Neighbor algorithm that would typically require using the scikit-learn
library. This program is specifically applied to the classic iris flower datasets problem.

I completed this using Jupyter Notebook. To run the program use the two iris-testing-data.csv and iris-training-data.csv files or feel 
free to use your own data so long as it's formatted the same way as my data.

This program takes in your source training and testing data outputs then ouputs its prediction for what kind of iris flower is being represneted. The output of the files looks like this: 

Grigor Sargsyan

#, True, Predicted
1,Iris-setosa,Iris-setosa
2,Iris-setosa,Iris-setosa
3,Iris-setosa,Iris-setosa
4,Iris-setosa,Iris-setosa
5,Iris-setosa,Iris-setosa
6,Iris-setosa,Iris-setosa
7,Iris-setosa,Iris-setosa
8,Iris-setosa,Iris-setosa
9,Iris-setosa,Iris-setosa
10,Iris-setosa,Iris-setosa
11,Iris-setosa,Iris-setosa
12,Iris-setosa,Iris-setosa
13,Iris-setosa,Iris-setosa
14,Iris-setosa,Iris-setosa
15,Iris-setosa,Iris-setosa
16,Iris-setosa,Iris-setosa
17,Iris-setosa,Iris-setosa
18,Iris-setosa,Iris-setosa
19,Iris-setosa,Iris-setosa
20,Iris-setosa,Iris-setosa
21,Iris-setosa,Iris-setosa
22,Iris-setosa,Iris-setosa
23,Iris-setosa,Iris-setosa
24,Iris-setosa,Iris-setosa
25,Iris-setosa,Iris-setosa
26,Iris-versicolor,Iris-versicolor
27,Iris-versicolor,Iris-versicolor
28,Iris-versicolor,Iris-versicolor
29,Iris-versicolor,Iris-versicolor
30,Iris-versicolor,Iris-versicolor
31,Iris-versicolor,Iris-versicolor
32,Iris-versicolor,Iris-versicolor
33,Iris-versicolor,Iris-versicolor
34,Iris-versicolor,Iris-versicolor
35,Iris-versicolor,Iris-versicolor
36,Iris-versicolor,Iris-versicolor
37,Iris-versicolor,Iris-versicolor
38,Iris-versicolor,Iris-versicolor
39,Iris-versicolor,Iris-versicolor
40,Iris-versicolor,Iris-versicolor
41,Iris-versicolor,Iris-versicolor
42,Iris-versicolor,Iris-versicolor
43,Iris-versicolor,Iris-versicolor
44,Iris-versicolor,Iris-versicolor
45,Iris-versicolor,Iris-versicolor
46,Iris-versicolor,Iris-virginica
47,Iris-versicolor,Iris-versicolor
48,Iris-versicolor,Iris-virginica
49,Iris-versicolor,Iris-versicolor
50,Iris-versicolor,Iris-versicolor
51,Iris-virginica,Iris-virginica
52,Iris-virginica,Iris-virginica
53,Iris-virginica,Iris-virginica
54,Iris-virginica,Iris-virginica
55,Iris-virginica,Iris-virginica
56,Iris-virginica,Iris-virginica
57,Iris-virginica,Iris-versicolor
58,Iris-virginica,Iris-virginica
59,Iris-virginica,Iris-virginica
60,Iris-virginica,Iris-virginica
61,Iris-virginica,Iris-virginica
62,Iris-virginica,Iris-virginica
63,Iris-virginica,Iris-virginica
64,Iris-virginica,Iris-virginica
65,Iris-virginica,Iris-virginica
66,Iris-virginica,Iris-virginica
67,Iris-virginica,Iris-virginica
68,Iris-virginica,Iris-virginica
69,Iris-virginica,Iris-virginica
70,Iris-virginica,Iris-versicolor
71,Iris-virginica,Iris-virginica
72,Iris-virginica,Iris-virginica
73,Iris-virginica,Iris-virginica
74,Iris-virginica,Iris-virginica
75,Iris-virginica,Iris-virginica
Accuracy: 94.67%
