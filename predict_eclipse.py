from skyfield import almanac
from skyfield import elementslib
from skyfield import eclipselib
from skyfield.api import Topos
from skyfield.api import load

from scipy.optimize import leastsq

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import date, timedelta

from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier


def void():
    raw_data = load('de422.bsp')
    astros = ['sun', 'moon', 'mercury', 'mars', 'venus', 'jupiter', 'saturn', 'neptune', 'uranus']
    periods = [365, 365, 88, 687, 225, 4333, 10759, 60190, 30687]
    earth = raw_data["earth"]
    pataliputra = earth + Topos("25.61 N", "85.16 E")
    ts = load.timescale()

    segment = raw_data.segments[0]
    start, end = segment.time_range(ts)

    earth = raw_data['earth']
    mars = raw_data['mars']
    sun = raw_data['sun']
    moon = raw_data['moon']
    tY0 = ts.utc(476, 1, 1)
    tY1 = ts.utc(485, 12, 31)

    # lunar eclipse 3
    timeTrueEclipse, trueEclipse, detailY = eclipselib.lunar_eclipses(tY0, tY1, raw_data)


    Ystart_date = date(476, 1, 1)
    Yend_date = date(485, 12, 31)
    Ytime_diff = (Yend_date - Ystart_date).days

    newEclipseDate = []
    for eclipseTime in timeTrueEclipse:
        eclipseYear = eclipseTime.tt_calendar()[0]
        eclipseMonth = eclipseTime.tt_calendar()[1]
        eclipseDay = eclipseTime.tt_calendar()[2]
        newEclipseTime = ts.utc(eclipseYear, eclipseMonth, eclipseDay)
        newEclipseDate.append(newEclipseTime)

    eclipseDateDict = {}
    for i in range(len(newEclipseDate)):
        eclipseDate = newEclipseDate[i]
        eclipseType = trueEclipse[i]
        eclipseDateDict[eclipseDate] = eclipseType

    totalDatesAnalyzed = []
    for day in range(Ytime_diff):
        currTime = Ystart_date + timedelta(day)
        currYear = currTime.year
        currMonth = currTime.month
        currDay = currTime.day
        currDate = ts.utc(currYear, currMonth, currDay)
        totalDatesAnalyzed.append(currDate)

    lunarEclipseValues = {}
    found = False
    for currDate in totalDatesAnalyzed:
        found = False
        for eclipseDate in newEclipseDate:
            if currDate == eclipseDate:
                lunarEclipseVal = eclipseDateDict[eclipseDate]
                lunarEclipseValues[currDate] = lunarEclipseVal
                found = True
            if not found:
                lunarEclipseValues[currDate] = 0
    trueEclipseFinal = list(lunarEclipseValues.values())

    predLunarThree = []
    predLunarFour = []

    predDm = []
    predDs = []
    predShadow = []

    predBetaThree = []
    predBetaFour = []
    predPhase = []

    for time in totalDatesAnalyzed:
        # Initial value calculation
        moonOsculatingi = elementslib.osculating_elements_of(moon.at(time))
        sunOsculatingi = elementslib.osculating_elements_of(sun.at(time))
        DMmi = moonOsculatingi.mean_motion_per_day.arcminutes()
        DMsi = sunOsculatingi.mean_motion_per_day.arcminutes()
        DMmi = DMmi % 21600
        DMsi = DMsi % 21600
        Dmi = (10 / 247) * DMmi
        Dsi = (11 / 20) * DMsi
        Dshadowi = (8 * DMmi - 25 * DMsi) / 60
        phase = almanac.fraction_illuminated(raw_data, 'moon', time)

        # Beta calculation
        lunarBetaThree = moonOsculatingi.argument_of_latitude.arcminutes()
        lunarBetaFour = moon.at(time).ecliptic_latlon()[1].arcminutes()

        # Obscured Portion Calculation (Proposal Formula)
        obscLunThree = (1 / 2) * (Dmi + Dshadowi) - lunarBetaThree
        obscLunFour = (1 / 2) * (Dmi + Dshadowi) - lunarBetaFour

        # Constructing Formula Dataset
        predLunarThree.append(obscLunThree)
        predLunarFour.append(obscLunFour)

        # Constructing Variable Datasets
        predBetaThree.append(lunarBetaThree)
        predBetaFour.append(lunarBetaFour)
        predDm.append(Dmi)
        predDs.append(Dsi)
        predShadow.append(Dshadowi)
        predPhase.append(phase)

    predEclipsesThree = []

    for i in range(len(predLunarThree)):
        obscuredPred = predLunarThree[i]
        dmPred = predDm[i]
        if obscuredPred < 0:
            predEclipsesThree.append(0)
            continue
        if obscuredPred < dmPred:
            predEclipsesThree.append(1)
            continue
        if obscuredPred >= dmPred:
            predEclipsesThree.append(2)
            continue

    predEclipsesFour = []

    for i in range(len(predLunarFour)):
        obscuredPred = predLunarFour[i]
        dmPred = predDm[i]
        if obscuredPred < 0:
            predEclipsesFour.append(0)
            continue
        if obscuredPred < dmPred:
            predEclipsesFour.append(1)
            continue
        if obscuredPred >= dmPred:
            predEclipsesFour.append(2)
            continue

    lunarEclipseDateTime = []
    for i in range(len(list(lunarEclipseValues.keys()))):
        time = list(lunarEclipseValues.keys())[i]
        lunarEclipseDateTime.append(time.utc_datetime())

    onlyEclipseDateTime = []
    for i in range(len(newEclipseDate)):
        time = newEclipseDate[i]
        onlyEclipseDateTime.append(time.utc_datetime())

    onlyEclipseVal = []
    onlyEclipseDateFinal = []
    for i in range(len(onlyEclipseDateTime)):
        eclipse = trueEclipse[i]
        time = onlyEclipseDateTime[i]
        if eclipse != 0:
            onlyEclipseVal.append(eclipse)
            onlyEclipseDateFinal.append(time)
        else:
            continue

    predtY0 = ts.utc(600, 1, 1)
    predtY1 = ts.utc(610, 1, 1)
    predtY0DT = date(600, 1, 1)
    predtY1DT = date(610, 1, 1)
    predtime_diff = (predtY1DT - predtY0DT).days

    timePredEclipse2, predEclipse2, predDetailY2 = eclipselib.lunar_eclipses(predtY0, predtY1, raw_data)
    predtime_diff

    predDates = []
    predStartDate = date(600, 1, 1)

    for day in range(predtime_diff):
        currTime = predStartDate + timedelta(day)
        currYear = currTime.year
        currMonth = currTime.month
        currDay = currTime.day
        currDate = ts.utc(currYear, currMonth, currDay)
        predDates.append(currDate)

    newEclipseDate2 = []
    for eclipseTime in timePredEclipse2:
        eclipseYear = eclipseTime.tt_calendar()[0]
        eclipseMonth = eclipseTime.tt_calendar()[1]
        eclipseDay = eclipseTime.tt_calendar()[2]
        newEclipseTime = ts.utc(eclipseYear, eclipseMonth, eclipseDay)
        newEclipseDate2.append(newEclipseTime)

    eclipseDateDict2 = {}
    for i in range(len(newEclipseDate2)):
        eclipseDate = newEclipseDate2[i]
        eclipseType = predEclipse2[i]
        eclipseDateDict2[eclipseDate] = eclipseType

    lunarEclipseValuesPred = {}
    found = False
    for currDate in predDates:
        found = False
        for eclipseDate in newEclipseDate2:
            if currDate == eclipseDate:
                lunarEclipseVal = eclipseDateDict2[eclipseDate]
                lunarEclipseValuesPred[currDate] = lunarEclipseVal
                found = True
            if not found:
                lunarEclipseValuesPred[currDate] = 0
    predEclipseFinal = list(lunarEclipseValuesPred.values())

    predLunar2 = []
    predDm2 = []
    predDs2 = []
    predShadow2 = []

    predBetaThree2 = []

    predLunarThree2 = []
    predPhase2 = []

    for time in predDates:
        # Initial value calculation
        moonOsculatingi = elementslib.osculating_elements_of(moon.at(time))
        sunOsculatingi = elementslib.osculating_elements_of(sun.at(time))
        DMmi = moonOsculatingi.mean_motion_per_day.arcminutes()
        DMsi = sunOsculatingi.mean_motion_per_day.arcminutes()
        DMmi = DMmi % 21600
        DMsi = DMsi % 21600
        Dmi = (10 / 247) * DMmi
        Dsi = (11 / 20) * DMsi
        Dshadowi = (8 * DMmi - 25 * DMsi) / 60
        phase = almanac.fraction_illuminated(raw_data, 'moon', time)
        lunarBetaThree = moonOsculatingi.argument_of_latitude.arcminutes()

        # Obscured Portion Calculation (Ancient Indian Formula)
        obscLunThree = (1 / 2) * (Dmi + Dshadowi) - lunarBetaThree

        # Constructing Formula Dataset
        predLunarThree2.append(obscLunThree)

        # Constructing Variable Datasets
        predBetaThree2.append(lunarBetaThree)
        predDm2.append(Dmi)
        predDs2.append(Dsi)
        predShadow2.append(Dshadowi)
        predPhase2.append(phase)

        v3Matrix = np.stack((predBetaFour, predDm, predShadow, predPhase), axis=-1)
        v3Matrix2 = np.stack((predBetaThree2, predDm2, predShadow2, predPhase2), axis=-1)

        gbc = GradientBoostingClassifier(random_state=0, n_estimators=1000, learning_rate=.015, subsample=.0099)
        gbc.fit(v3Matrix, trueEclipseFinal)
        predGBC = gbc.predict(v3Matrix2)
        gbc.score(v3Matrix, trueEclipseFinal), accuracy_score(predEclipseFinal, predGBC)

        trueCount = 0
        for i in trueEclipseFinal:
            if i > 0:
                trueCount += 1
        trueProp = trueCount / len(trueEclipseFinal)

        ancIndCount = 0
        for i in predEclipsesFour:
            if i > 0:
                ancIndCount += 1
        ancIndProp = ancIndCount / len(predEclipsesThree)

        gbcCount = 0
        for i in predGBC:
            if i > 0:
                gbcCount += 1
        gbcProp = gbcCount / len(predGBC)

        method = ["True", "Ancient Indian", "Gradient Boosting"]
        proportion = [trueProp, ancIndProp, gbcProp]
        count = [trueCount, ancIndCount, gbcCount]
        propDF = pd.DataFrame({"method": method, "proportion": proportion, "count": count})

        distances = []
        dist = 0

        for i in range(len(onlyEclipseVal) - 1):
            dist = (onlyEclipseDateFinal[i + 1] - onlyEclipseDateFinal[i]).days
            distances.append(dist)

        distHalf = []
        for i in range(len(distances)):
            dist = distances[i]
            newDist = dist * (.5)
            distHalf.append(newDist)

        midpoints = []
        midptDate = []
        dist = 0
        for i in range(len(distances)):
            if i % 2 == 0:
                newDate = onlyEclipseDateFinal[i] + timedelta(distHalf[i])
                newPoint = onlyEclipseVal[i]
                midpoints.append(newPoint)
                midptDate.append(newDate)

        count = 0
        meanDist = 0
        for i in distHalf:
            if i < 110:
                count += 1
                meanDist += i

        meanDist = meanDist / count
        meanDistArr = []
        meanDistArrNoTime = []
        for i in range(len(midpoints)):
            meanDistArr.append(timedelta(meanDist * 2))
            meanDistArrNoTime.append(meanDist)

        meanGuess = 0.5
        stdGuess = .5
        phaseGuess = 3
        freqGuess = 6.03
        ampGuess = 0.5
        length = len(predPhase[:365])
        linSpace = np.linspace(0, 4 * np.pi, length)
        data = predPhase[:365]

        optimizer = lambda x: x[0] * np.sin(x[1] * linSpace + x[2]) + x[3] - data
        ampPred, freqPred, phasePred, meanPred = leastsq(optimizer, [ampGuess, freqGuess, phaseGuess, meanGuess])[0]
        xPred = np.arange(0, max(linSpace) + 5, 0.1)
        fitted = ampPred * np.sin(freqPred * xPred + phasePred) + meanPred

        plt.plot(linSpace, data, 'o')
        plt.plot(xPred, fitted, label='Fitted Model')
        plt.legend(loc="upper right")
        plt.title("Phases of the Moon: Sinusoidal Model")
        plt.show()

    newErrVal = (meanDist/354.5) * 6.28318531
    newErr = [newErrVal, newErrVal, newErrVal]

    meanGuessEcl = 1.5
    stdGuessEcl = .5
    phaseGuessEcl = 3.5
    freqGuessEcl = 0.6
    ampGuessEcl = 1

    lengthEcl = len(midpoints[:3])
    linSpaceEcl = np.linspace(0, 4*np.pi, 3)
    dataEcl = midpoints[:3]
    linEcl = np.linspace(0, 4*np.pi, 9)

    lengthPred = 5
    linSpacePred = np.linspace(0, 8*np.pi, N3)
    predVisualize = [1, 2, 1, 2, 1]

    optimizerEcl = lambda x: x[0]*np.sin(x[1]*linEcl+x[2]) + x[3] - midpoints
    ampPredEcl, freqPredEcl, phasePredEcl, meanPredEcl = leastsq(optimizerEcl, [ampGuessEcl, freqGuessEcl, phaseGuessEcl, meanGuessEcl])[0]
    xPredEcl = np.arange(-5,max(linEcl) + 15,0.1)
    fittedEcl =(5.3 * ampPredEcl * np.sin((.69) * freqPredEcl * xPredEcl + phasePredEcl - 8.05) + meanPredEcl + 0.178)

    Err3 = [1.5691716204381758, 1.5691716204381758, 1.5691716204381758, 1.5691716204381758,
            1.5691716204381758]

    plt.errorbar(x= linSpacePred, y= predVisualize, xerr = Err3, fmt= "o", color= "red", ecolor="red", label="Predicted Eclipse")
    plt.errorbar(x= linSpaceEcl, y= dataEcl, xerr = newErr, fmt= "o", ecolor="blue")
    plt.plot(xPredEcl, fittedEcl, label="Fitted Model")
    plt.legend(loc= "upper right")
    plt.title("Lunar Eclipses: Sinusoidal Model")
    plt.show()