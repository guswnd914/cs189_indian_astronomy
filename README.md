# Project S Final: Team VROK

**Team Members:**  Vikash Raja, Musa Jalis, Otto Kwon, Anika Rede

**Navigating the Repository:** 

In the scripts folder, we have a few files that contain parts of our model. To make it easier to run, we've ordered the predictions into two files:

1. predict.py handles prediction of planetary motion. Calling this file will take in the dataset from the time period of 450 to 550 A.D., around the time the ancient Indian astronomical models were created, clena the code, and retrieve the parameters we needed including distance, elliptical latitude and longitude, and illuminated fraction on the moon. Once the data is processed, it will save to our data folder. Then the epicyclic planetary motion model is generated, set with the proper parameters, trained with Fourier series, and evaluated. The results for prediction and evaluation are saved to the data folder as well. [ADD MORE DETAILS]

2. predict_eclipse.py will create the eclipse model. First we go through the dataset from the skyfield API and generate the true values for lunar eclipses and moon phases. This data is used later on to estimate accuracy of our results. Then, we find the parameters from our data in variables like osculating elements and arguments of latitude, and construct the formulas from our mathematical model. We use the true values to understand MSE and accuracy of our results in count of eclipses. [ADD MORE DETAILS]
