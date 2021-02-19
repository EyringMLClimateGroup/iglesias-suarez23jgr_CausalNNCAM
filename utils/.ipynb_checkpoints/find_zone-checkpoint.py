#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
"""
HISTORY:
07-Oct-2019 Adapted to python3,                FI
24-May-2018 Find exact location,               FI
02-Oct-2015 __name__ fixed,                    FI
01-Oct-2015 Written,                           FI

DESCRIPTION:
Find nearest region (e.g. lat or lon).

NOTES:
To see some examples run it as main script (i.e.
from the terminal: python find_zone.py).

East longitudes must be given in positive values
(e.g. From 0. to 180.; lon1=0 & lon2=180).

West longitudes must be given in negative values
(e.g. From -180. to -0.5; lon1=-180 & lon2=-0.5).

Note that is important the order of lon1 & lon2.
East values from 0 to 180; West values from -180 to -0.5

TO-DO:
Fix/find solution for the cases when the zone goes from west
to east (e.g. 10W-35E).

"""





######################### PACKAGES #########################

# System
import sys; sys.dont_write_bytecode = True
import os, traceback, optparse
import time
import re
import pdb

# Python maths and stats
import numpy as np

# Analysis setting

# Data

# Plotting

# Utils

############################################################





######################### OPTIONS ##########################

## SCRIPT
#
# Version
vScript = '(__name__ - fixed): 1.1'
# Name
nScript = 'find_zone.py'

###########################################################




######################## DEFINITIONS #######################

def main():

    # Make global the options passed to the script
    '''
    python find_zone.py [-h,--help] [-v,--verbose] [--version]
    '''
    global options, args


def find_zone(longitudes, lon1, lon2):
    
    '''
    Takes longitudes array (0-360) and lon. coordinates.
    Returns indexes between given lon. coordinates.
    '''

    # Convert lons to 0-360 coordinates
    Lon1 = [360 + int(lon1),int(lon1)][lon1 >= 0]
    Lon2 = [360 + int(lon2),int(lon2)][lon2 >= 0]

    # Longitudes minus the coordinates
    n1 = [np.array(i - Lon1) for i in longitudes]
    n2 = [np.array(i - Lon2) for i in longitudes]

    # Find the minimum value of the made-up arrays
    idx1 = abs(np.array(n1)).argmin()

    if abs(lon1) == abs(lon2):
        if lon1 == 0 and lon2 == 0:
            idx2 = len(longitudes) - idx1
        else:
            idx2 = len(longitudes) - idx1
    else:
        idx2 = abs(np.array(n2)).argmin() + 1

    # Return indexes
    return idx1, idx2

def find_zone1(longitudes, lon1, lon2='None'):
    
    '''
    Takes longitudes array (0-360) and lon. coordinates.
    Returns indexes between given lon. coordinates.
    '''

    # Convert lons to 0-360 coordinates
    Lon1 = [360 + int(lon1),int(lon1)][lon1 >= 0]
    if lon2 != 'None':
        Lon2 = [360 + int(lon2),int(lon2)][lon2 >= 0]

    # Longitudes minus the coordinates
    n1 = [np.array(i - Lon1) for i in longitudes]
    if lon2 != 'None':
        n2 = [np.array(i - Lon2) for i in longitudes]

    # Find the minimum value of the made-up arrays
    idx1 = abs(np.array(n1)).argmin()
    if lon2 != 'None':
        if abs(lon1) == abs(lon2):
            if lon1 == 0 and lon2 == 0:
                idx2 = len(longitudes) - idx1
            else:
                idx2 = len(longitudes) - idx1
        else:
            idx2 = abs(np.array(n2)).argmin() + 1

    # Return indexes
    if lon2 != 'None':
        return idx1, idx2
    else:
        return idx1

############################################################





########################### MAIN ###########################

if __name__ == '__main__':

    ## EXAMPLES
    #
    
    # Made-up longitudes
    lons = np.array([0. ,2.5, 5. ,7.5,10.,   12.5,   15. ,   17.5,
     20. ,   22.5,   25. ,   27.5,   30. ,   32.5,   35. ,   37.5,
     40. ,   42.5,   45. ,   47.5,   50. ,   52.5,   55. ,   57.5,
     60. ,   62.5,   65. ,   67.5,   70. ,   72.5,   75. ,   77.5,
     80. ,   82.5,   85. ,   87.5,   90. ,   92.5,   95. ,   97.5,
    100. ,  102.5,  105. ,  107.5,  110. ,  112.5,  115. ,  117.5,
    120. ,  122.5,  125. ,  127.5,  130. ,  132.5,  135. ,  137.5,
    140. ,  142.5,  145. ,  147.5,  150. ,  152.5,  155. ,  157.5,
    160. ,  162.5,  165. ,  167.5,  170. ,  172.5,  175. ,  177.5,
    180. ,  182.5,  185. ,  187.5,  190. ,  192.5,  195. ,  197.5,
    200. ,  202.5,  205. ,  207.5,  210. ,  212.5,  215. ,  217.5,
    220. ,  222.5,  225. ,  227.5,  230. ,  232.5,  235. ,  237.5,
    240. ,  242.5,  245. ,  247.5,  250. ,  252.5,  255. ,  257.5,
    260. ,  262.5,  265. ,  267.5,  270. ,  272.5,  275. ,  277.5,
    280. ,  282.5,  285. ,  287.5,  290. ,  292.5,  295. ,  297.5,
    300. ,  302.5,  305. ,  307.5,  310. ,  312.5,  315. ,  317.5,
    320. ,  322.5,  325. ,  327.5,  330. ,  332.5,  335. ,  337.5,
    340. ,  342.5,  345. ,  347.5,  350. ,  352.5,  355. ,  357.5])

    real_lons = ['0E', '2.5E', '5E', '7.5E', '10E', '12.5E', '15E',
                 '17.5E', '20E', '22.5E', '25E', '27.5E', '30E', '32.5E',
                 '35E', '37.5E', '40E', '42.5E', '45E', '47.5E', '50E',
                 '52.5E', '55E', '57.5E', '60E', '62.5E', '65E', '67.5E',
                 '70E', '72.5E', '75E', '77.5E', '80E', '82.5E', '85E',
                 '87.5E', '90E', '92.5E', '95E', '97.5E', '100E', '102.5E',
                 '105E', '107.5E', '110E', '112.5E', '115E', '117.5E',
                 '120E', '122.5E', '125E', '127.5E', '130E', '132.5E',
                 '135E', '137.5E', '140E', '142.5E', '145E', '147.5E',
                 '150E', '152.5E', '155E', '157.5E', '160E', '162.5E',
                 '165E', '167.5E', '170E', '172.5E', '175E', '177.5E',
                 '180E', '177.5W', '175W', '172.5W', '170W', '167.5W',
                 '165W', '162.5W', '160W', '157.5W', '155W', '152.5W',
                 '150W', '147.5W', '145W', '142.5W', '140W', '137.5W',
                 '135W', '132.5W', '130W', '127.5W', '125W', '122.5W',
                 '120W', '117.5W', '115W', '112.5W', '110W', '107.5W',
                 '105W', '102.5W', '100W', '97.5W', '95W', '92.5W', '90W',
                 '87.5W', '85W', '82.5W', '80W', '77.5W', '75W', '72.5W',
                 '70W', '67.5W', '65W', '62.5W', '60W', '57.5W', '55W',
                 '52.5W', '50W', '47.5W', '45W', '42.5W', '40W', '37.5W',
                 '35W', '32.5W', '30W', '27.5W', '25W', '22.5W', '20W',
                 '17.5W', '15W', '12.5W', '10W', '7.5W', '5W', '2.5W']

    
    # Find zone: 0(0E) to 100 (100E)
    lon1, lon2 = find_zone(lons, 0, 100)
    # Result
    print ("Test 1, for: 0(0E) & 100(100E)")
    print ('lons (indexes):'); print (lons[lon1:lon2])
    print ('real_lons (values)'); print (real_lons[lon1:lon2])

     
    # Find zone: -170(170W) to -120 (120W)
    lon1, lon2 = find_zone(lons, -170, -120)
    # Result
    print ("Test 2, for: -170(170W) & -120(120W)")
    print ('lons (indexes):'); print (lons[lon1:lon2])
    print ('real_lons (values)'); print (real_lons[lon1:lon2])

    # Find zone: 100(100E) to -170 (170W)
    lon1, lon2 = find_zone(lons, 100, -170)
    # Result
    print ("Test 3, for: 100(100E) & -170(170W)")
    print ('lons (indexes):'); print (lons[lon1:lon2])
    print ('real_lons (values)'); print (real_lons[lon1:lon2])
    

###########################################################


