# Ancient Indian's Pulsating Epicycle Model and Predictions

**Team Members:**  Vikash Raja, Musa Jalis, Otto Kwon, Anika Rede

This project allows to predict motions of astronomical objects using `Sūrya-siddhānta` planetary model date back to 500 CE integrating it with machine learning algorithm. 

The model is subdivided into two categories : `Planetary Motion` and `Lunar Phases / Eclipses`.

## Ancient Indian's Astronomy Model

* Ancient Indians believed that the planets, sun, and moon were circularly orbiting around Earth in epicylical manner where angular velocity expands and contracts depending on each quadrant.
* The model can be characterized by the equations :




## Dataset Used
| API | File Name | Concentration | Referring Dates | Data Format | Size |
| ------------- | -------- | --------- | --------- | -------------- | ----- | 
| Skyfield | de422 | Astronomy  | -3000 BCE - 3000 BCE | BSP | 623 MB |


## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* Install `<Python 3.5>` or above version.
* Run following command on your terminal.
```
pip install -r required_packages.txt
```

## Navigating the Repository:

```bash
├── scripts
│   ├── predict.py
│   ├── predict_eclipse.py
│   ├── predict_epicycle.py
│   ├── predict_tool.py
│   ├── process_data.py
│   └── visualize.py
├── data
├── evaluation
└── visualization
```

* `scripts`: A directory for necessary python scripts for the model.
  
  * `process_data.py:` 
    * data gathering / processing script module
    * load Skyfield data ⇒ collect necessary model features ⇒ convert feature data types ⇒ save at `data` directory in json format.

  * `predict_epicycle.py:` 
    * data cleaning and data modelling <i>(planetary movement prediction)</i> script module.
    * build `DataArrangeTool` class to simplify data fetching process.
      - load processed data from `data` directory
      - do feature engineering (i.e. [distance, θ] → [distance_x, distance_y])
      - append them to its attributes
      - save it in `data` directory.
    * create inheritance `PulsatingEpicycleModel` class to proceed with Fast Fourier Transformation and regression.
      - split train/test data varying to test_size input.
      - generate samples of training data and testing data varying to number of data points and sigma.
      - transform the training data by by FFT frequency and apply curve-fit objected by sin function.
      - predict distance by fitting it test data and store the results in list.
      - display optimal parameters for model and calculate evalulation based on several metrics (ie. MSE)
   
  * `predict_eclipse.py`
    * data modelling <i>(lunar eclipise / phases)</i> script module.
       - perform sinusoidal regression on 
      
      
