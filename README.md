INTRODUCTION

The problem of access to clean water by urban poor households in low-income countries persists because the households are not connected to water distribution network (WDN). Hence they lack public-provided water supply in a poor sanitary environment, and are liable to suffer water-related diseases. This is especially true of the people of Nyanya-Mararaba Town in Nigeria’s Federal Capital Territory (Dada, 2024; Olokesusi et al., 2018). They seek water from different sources such as well, water vendors and borehole, and so they pay heavily for clean water, which is provided by private businesses (Chukwu, 2015; UNDP, 2015). Since urban poor households are not connected to public-provided WDN, there is no water meter and so no automatic means of gathering empirical data about the volume of water they use. Thus, it is difficult to evaluate volume of water needed for planning water supply to the households. Hence, this study aims to provide decision-makers with a tool to evaluate volume of water needed by urban poor households for planning water supply to places where there is no WDN. 

THEORETICAL BACKGROUND

Geospatial Technology and Machine Learning Techniques

It is important to evaluate volume of water used in order to discover whether the households have access to adequate water supply to meet WASH standards.   The accuracy of the evaluation can be improved by combining geospatial technology (GST) and machine learning techniques (MLT). Geospatial data or geospatial features (GSF) in machine learning models are geographical variables or features that represent real-world entities that are being modelled. GSF are important in machine learning modeling because they provide additional context and information that can improve the accuracy of models. 

GST is a collective term for the various modern technologies that help in collection, analysis and interpretation of geospatial data (Reed and Ritz, 2003). GST includes remote sensing (RS), Global Positioning System (GPS) and Geospatial Information System (GIS). GST, which represents the physical world in a virtual environment, is used in modelling, simulation and visualization of geospatial information, which can inform both present and future decisions. GPS and GIS systems are used in this study to collect and process geospatial data respectively, and then they are integrated with MLT to model water use.

MLT are coded algorithms which represent mathematical or statistical models in the computer. Four ML models – Multilinear regression (MLR), random forest (RF), support vector regression (SVR) and artificial neural network (ANN) – were employed in this study. The four are chosen because they represent MLTs in the supervised learning category, and it has been shown that ANN-based techniques outperform conventional techniques and provide effective solutions for many geospatial data analysis (Kiwelekar et al., 2020).

Multilinear Regression (MLR)

Linear regression is a technique for investigating the relationship between an independent variable or feature and a dependent variable or outcome. Linear regression model describes the relationship between a dependent variable y and independent variables x with a straight line that is defined by Equation (1):

$y^p= \theta_1x+ \theta_o$	                                                                                                                                                           (1)

In this expression, y is the vector of the response values. The x symbol describes the matrix of features which the algorithm uses to predict the y vector. x is a matrix that contains only numeric values. o and 1 are parameters that the linear regression uses to create the prediction, yp.

For nine features the space is nine-dimensional and the regression Equation becomes Equation (2).

$y^p = \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + \theta_4x_4 + \theta_5x_5 + \theta_6x_6 + \theta_7x_7 + \theta_8x_8 + \theta_9x_9 + \theta_o$		                               (2)

Equation (2) can be extended to form polynomials by creating new features from the variables x1… x9. Thus, second degree quadratic, third-degree cubic, fourth-degree quartic and fifth-degree quintic polynomials can be expressed respectively as follows:

$y_2^p = \theta_1x_1^2 + \theta_2x_2^2 + \theta_3x_3^2 + \theta_4x_4^2 + \theta_5x_1x_2 + \theta_6x_3x_4 + \theta_7x_1x_3 + \theta_8x_1 + \theta_9x_2 + \theta_{10}x_3 + \theta_{11}x_4 + \theta_{12}x_5 + \theta_{13}x_6 + \theta_{14}x_7 + \theta_{15}x_8 + \theta_{16}x_9 + \theta_o$		                                                        (3)

$y_3^p = \theta_1x_1^3 + \theta_2x_2^3 + \theta_3x_3^3 + \theta_4x_4^3 + \theta_5x_1^2 + \theta_6x_2^2 + \theta_7x_3^2 + \theta_8x_4^2 + \theta_9x_1x_2 + \theta_{10}x_3x_4 + \theta_{11}x_1x_3 + \theta_{12}x_1 + \theta_{13}x_2 + \theta_{14}x_3 + \theta_{15}x_4 + \theta_{16}x_5 + \theta_{17}x_6 + \theta_{18}x_7 + \theta_{19}x_8 + \theta_{20}x_9 + \theta_o$	                                                      (4)

$y_4^p = \theta_1x_1^4 + \theta_2x_2^4 + \theta_3x_3^4 + \theta_4x_4^4 + \theta_5x_1^3 + \theta_6x_2^3 + \theta_7x_3^3 + \theta_8x_4^3 + \theta_9x_1^2 + \theta_{10}x_2^2 + \theta_{11}x_3^2 + \theta_{12}x_4^2 + \theta_{13}x_1x_2 + \theta_{14}x_3x_4 + \theta_{15}x_1x_3 + \theta_{16}x_1 + \theta_{17}x_2 + \theta_{18}x_3 + \theta_{19}x_4 + \theta_{20}x_5 + \theta_{21}x_6 + \theta_{22}x_7 + \theta_{23}x_8 + \theta_{24}x_9 + \theta_o$												    (5)

$y_5^p = \theta_1x_1^5 + \theta_2x_2^5 + \theta_3x_3^5 + \theta_4x_4^5 + \theta_5x_1^4 + \theta_6x_2^4 + \theta_7x_3^4 + \theta_8x_4^4 + \theta_9x_1^3 + \theta_{10}x_2^3 + \theta_{11}x_3^3 + \theta_{12}x_4^3 + \theta_{13}x_1^2 + \theta_{14}x_2^2 + \theta_{15}x_3^2 + \theta_{16}x_4^2 + \theta_{17}x_1x_2 + \theta_{18}x_3x_4 + \theta_{19}x_1x_3 + \theta_{20}x_1 + \theta_{21}x_2 + \theta_{22}x_3 + \theta_{23}x_4 + \theta_{24}x_5 + \theta_{25}x_6 + \theta_{26}x_7 + \theta_{27}x_8 + \theta_{28}x_9 + \theta_o$	(6)

MATERIALS AND METHODS

Study Area

The growing population of Nigeria’s Federal Capital City is concentrated at Nyanya-Mararaba Town where an estimated 70% of the people working in the city live. Covering an area of 14 square kilometers, Nyanya-Mararaba is a peri-urban town consisting of informal dwellings where many poor people find shelter. 

Datasets

This study use Nyanya-Mararaba datasets described in Taiwo et al., (2023). 

Nyanya-Mararaba Dataset

There are twelve wards in Nyanya-Mararaba Town. A total of 1200 households were interviewed; 100 respondents were randomly selected from each ward. Each respondent, as a representative of a poor household, was given an identity number from 1 to 1200. 

Dry and Wet Seasons Datasets

Volume of water used per day for dry and wet seasons, represented by September to February and March to August respectively. A dataset of 1200 records was derived for each season of the year, resulting in two datasets, one for dry season and another for wet season. 

Selection of Explanatory Variables

Explanatory variables are features that result in optimal model performance. Feature selection techniques measure the dependency between independent variables and dependent variable. The more the dependency, the more important the feature. In this research, explanatory variables were selected using Pearson Correlation Coefficient, which has acceptable scores between -3.0 and -9.0 for negative correlation; +3.0 and +9.0 for positive correlation (Schober, Boer and Schwarte, 2018). The correlation between the variables are given in Table 1, in which the correlation values between the explanatory variables (independent variables) and volume (dependent variable) are formatted in bold. 

Modeling Experiments

The experiments were carried out with Python codes in JupyterLab®. Using a pipeline in Scikit Learn® library, the instance of each model was infused with a scaler, which normalizes the dataset so that all the data have the same scale (Pedregosa et al., 2012).  Thus, an empty model was created. The model was trained or fitted with a training dataset. Training enables the model to learn the relationship between the explanatory variables and the target variable. After training, the model was ready for predictions. The results were evaluated. If the resulting errors were not within tolerance (0.70 ≤ R2 ≥ 0.99), then the model needed refining through hyper parameter tuning and it should be built again. If the errors were within tolerance, the model is built with the final hyper parameter value. 

Coding the Models 

The complete dataset was first one-hot encoded before it was split into training and test datasets.  The MLR, RF, SVR and ANN models, which were named model in the codes, were each created as an instance of their respective classes in Scikit Learn® library: SGDRegressor, RandomForestRegressor, LinearSVR and MLPRegressor, which were each infused with a scaler to scale the dataset. model was fitted to the training dataset in order to learn the data. Two functions named training and testing were coded to carry out prediction with training dataset and test dataset respectively. 

MLR models actual and predicted volume of water using Equation (7) and Equation (8) respectively:

$Y_o^{(i)}= \theta_1\mathrm{I}(i) + \theta_2\mathrm{S}(i) + \theta_3\mathrm{R}(i) + \theta_4\mathrm{T}(i) + \theta_5\mathrm{t}(i)+ \theta_6\mathrm{A}(i) + \theta_7\mathrm{W}(i) + \theta_8\mathrm{d}(i) + \theta_9\mathrm{h}(i) + \theta_o$		(7)

$Y_p^{(i)} = \theta_1\mathrm{I}(i) + \theta_2\mathrm{S}(i) + \theta_3\mathrm{R}(i) + \theta_4\mathrm{T}(i) + \theta_5\mathrm{t}(i)+ \theta_6\mathrm{A}(i) + \theta_7\mathrm{W}(i) + \theta_8\mathrm{d}(i) + \theta_9\mathrm{h}(i) + \theta_o$		(8)

where i = (1, …, 1200) for 1200 examples or households; $Y_o$ is the actual volume of water used by a household; $Y_p$ is the predicted volume of water used by a household; $\theta$ is the parameters; 
$\theta_o$ is the bias or y-intercept; $\mathrm{I, S, R, T, t, A, W, d, and h}$ are the selected explanatory variables. 

Coding the Polynomials
Polynomials of higher degrees – quadratic, cubic, quartic and quintic polynomials as Equation (12), Equation (13), Equation (14) and Equation (15) respectively – were coded and tested for better accuracies than linear models. 


$Y = \theta_o + \theta_1\mathrm{I} + \theta_2\mathrm{S} + \theta_3\mathrm{R} + \theta_4\mathrm{T} + \theta_5\mathrm{t} + \theta_6\mathrm{A} + \theta_7\mathrm{W} + \theta_8 \mathrm{d} + \theta_9\mathrm{h} + \theta_{10}t\mathrm{A} + \theta_{11}\mathrm{Wd} + \theta_{12}\mathrm{td} + \theta_{13}\mathrm{t}^2 + \theta_{14}\mathrm{A}^2 + \theta_{15}\mathrm{W}^2 + \theta_{16}\mathrm{Ad}^2$		                                (10)

$Y = \theta_o + \theta_1\mathrm{I} + \theta_2\mathrm{S} + \theta_3\mathrm{R} + \theta_4\mathrm{T} + \theta_5\mathrm{t} + \theta_6\mathrm{A} + \theta_7\mathrm{W} + \theta_8\mathrm{d} + \theta_9\mathrm{h} + \theta_{10}\mathrm{tA} + \theta_{11}\mathrm{Wd} + \theta_{12}\mathrm{td} + \theta_{13}\mathrm{t}^2 + \theta_{14}\mathrm{A}^2 + \theta_{15}\mathrm{W}^2 + \theta_{16}\mathrm{d}^2 + \theta_{17}\mathrm{t}^3 + \theta_{18}\mathrm{A}^3 + \theta_{19}\mathrm{W}^3 + \theta_{20}\mathrm{d}^3$                                                                (11)

$Y = \theta_o + \theta_1\mathrm{I} + \theta_2\mathrm{S} + \theta_3\mathrm{R} + \theta_4\mathrm{T} + \theta_5\mathrm{t} + \theta_6\mathrm{A} + \theta_7\mathrm{W} + \theta_8\mathrm{d} + \theta_9\mathrm{h} + \theta_{10}\mathrm{tA} + \theta_{11}\mathrm{Wd} + \theta_{12}\mathrm{td} + \theta_{13}\mathrm{t}^2 + \theta_{14}\mathrm{A}^2 + \theta_{15}\mathrm{W}^2 + \theta_{16}\mathrm{d}^2 + \theta_{17}\mathrm{t}^3 + \theta_{18}\mathrm{A}^3 + \theta_{19}\mathrm{W}^3 + \theta_{20}\mathrm{d}^3 + \theta_{21}\mathrm{t}^4 + \theta_{22}\mathrm{A}^4 + \theta_{23}\mathrm{W}^4 + \theta_{24}\mathrm{d}^4$	                                                                    (12)

$Y = \theta_o + \theta_1\mathrm{I} + \theta_2\mathrm{S} + \theta_3\mathrm{R} + \theta_4\mathrm{T} + \theta_5\mathrm{t} + \theta_6\mathrm{A} + \theta_7\mathrm{W} + \theta_8\mathrm{d} + \theta_9\mathrm{h} + \theta_{10}\mathrm{tA} + \theta_{11}\mathrm{Wd} + \theta_{12}\mathrm{td} + \theta_{13}\mathrm{t}^2 + \theta_{14}\mathrm{A}^2 + \theta_{15}\mathrm{W}^2 + \theta_{16}\mathrm{d}^2 + \theta_{17}\mathrm{t}^3 + \theta_{18}\mathrm{A}^3 + \theta_{19}\mathrm{W}^3 + \theta_{20}\mathrm{d}^3 + \theta_{21}\mathrm{t}^4 + \theta_{22}\mathrm{A}^4 + \theta_{23}\mathrm{W}^4 + \theta_{24}\mathrm{d}^4 + \theta_{25}\mathrm{t}^5 + \theta_{26}\mathrm{A}^5 + \theta_{27}\mathrm{W}^5 + \theta_{28}\mathrm{d}^5$							                                  (14)

Coding the Model Performance Metrics

Three statistical metrics were used to measure the performance of the models: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R squared. 

RESULTS

The results of the experiments carried out in this research are presented and discussed in this section. 

Model Performances in Dry and Wet Seasons

The model performances were evaluated with dry and wet season datasets. The results are shown in Table 2 and 3. RF model performed better than other models during training. The models performed better with wet season data than dry season data. 
The high R2 values show that the model well learned the data during training and testing. 

Average Volume of Water Consumed by the Households
The actual and predicted average volume of water consumed per day by the households is given in Table 4. Average volume of water consumed per day by the households in dry season is 147.35L, while average volume of water consumed per day in wet season is 167.35L. This suggests that poor households in the study area consume more water in wet season. An explanation could be that they practice rain water harvesting, and so they have access to more clean water in wet season than in dry season. 

Validation of Model Performances

The validation of model performances is highlighted in Table 5 in the seasons. Visual inspection reveals that the model performance during validation is comparable to the performances during training and testing. Thus, the performances of the models were validated.  

The polynomials were validated and compared to the linear models. The third-degree polynomial gives better accuracies as a functional model to be used for evaluating volume of water consumed in urban poor households. It can also be seen in Figure 2 that the models accuracy increases with increasing number of features. But the accuracy gets to a point where it begins to decrease as more features were added. This confirms what researchers have observed as one of the characteristics of ML modeling (Bellman, 2003; Pires & Branco, 2019; Ahmad & Nassif, 2022). 

Functional Models for Predicting Household Water Consumption

Based on the above modeling and evaluation, functional models that can be used to predict volume of water consumed in dry and wet seasons in any urban poor households were developed. Parameters (that is, intercept and coefficients of the nine variables) were substituted into Equation (7) to give Equation (15) and Equation (16) respectively. Since third-degree cubic polynomial gave better accuracies, values of the parameters in dry and wet seasons were determined and substituted into the third-degree cubic polynomial model in Equation (13) to form Equation (17) and Equation (18) respectively, which are the functional models for predicting volume of water consumed in urban poor households in dry and wet seasons. 

$Y_{dry}^P = 162 + 0.0003\mathrm{I} + 5.39\mathrm{S} + 0.331\mathrm{R} + 1.80\mathrm{T} - 2.01\mathrm{t} - 0.0003\mathrm{A} + 0.0804\mathrm{W} + 0.0142\mathrm{d} - 0.0094\mathrm{h}$    						                                                                                    (15)

$Y_{wet}^P = 15.4 + 0.0003\mathrm{I} + 5.24\mathrm{S} + 0.108\mathrm{R} + 4.43\mathrm{T} - 2.03\mathrm{t} + 0.0003\mathrm{A} + 0.0495\mathrm{W} + 0.0012\mathrm{d} - 0.007\mathrm{h}$													                                                                                (16)

$Y_{dry}^P = -524000 + 0.000285\mathrm{I} + 6.54\mathrm{S} + 1.35\mathrm{R} + 18000\mathrm{T} + 2.851\mathrm{t} + 0.0296\mathrm{A} + 0.0475\mathrm{W} + 0.0928\mathrm{d} + 0.1110.007\mathrm{h} - 0.001320.007\mathrm{tA} + 0.000001980.007\mathrm{Wd} - 0.0008310.007\mathrm{td} - 0.110.007\mathrm{t}^2 - 0.0001410.007\mathrm{A}^2 - 0.0002630.007\mathrm{W}^2 + 0.000002730.007\mathrm{d}^2 + 0.0008740.007\mathrm{t}^3 + 0.0000001240.007\mathrm{A}^3 + 0.0000002010.007\mathrm{W}^3 - 0.00000001160.007\mathrm{d}^3$										                            (17)

$Y_{wet}^P = -649000 + 0.000304\mathrm{I} + 6.35\mathrm{S} + 1.07\mathrm{R} + 0.000223\mathrm{T} + 2.451\mathrm{t} - 0.0104\mathrm{A} + 0.0611\mathrm{W} + 0.0217\mathrm{d} + 0.2\mathrm{h} - 0.000968\mathrm{tA} + 0.0000436\mathrm{Wd} - 0.000499\mathrm{td} - 0.106\mathrm{t}^2 - 0.0000097\mathrm{A}^2 - 0.000508\mathrm{W}^2 + 0.0000638\mathrm{d}^2 + 0.000849\mathrm{t}^3 + 0.0000000345\mathrm{A}^3 + 0.000000401\mathrm{W}^3  - 0.0000000606\mathrm{d}^3$											                                            (18)

where
$Y_{dry}^P$ is the predicted volume of water consumed in dry season; $Y_{wet}^P$ is the predicted volume of water consumed in wet season; and \mathrm{I, S, R, T, t, A, W, d, h} are the explanatory variables. 

Thus, if the values of the explanatory variables are known for any households, the functional models can be used to determine volume of water consumed (in liter per day) by the households in dry season and wet season respectively. Since planning water supply requires review of water consumption and projection to the future (Charlesworth et al., 2020), these functional models are useful for planning water supply to households where there is no WDN.

CONCLUSION 

This study explores the combination of geospatial technology and machine learning techniques to develop functional models for predicting volume of water consumed by urban poor households. The functional models were validated with RMSE of 5.67L and 4.92L in dry and wet seasons respectively, so providing a tool for planning water supply to urban poor households where there is no WDN. 


