'''


@inproceedings{tirunagari2016automatic,
   title={Automatic detection of acute kidney injury episodes from primary care data},
   author={Tirunagari, Santosh and Bull, Simon C and Vehtari, Aki and Farmer, Christopher and de Lusignan, Simon and Poh,    Norman},
   booktitle={Computational Intelligence (SSCI), 2016 IEEE Symposium Series on},
   pages={1--6},
   year={2016},
   organization={IEEE}
}



'''

import numpy as np
import matplotlib.pyplot as plt

# start SAKIDA Algorithm
# Append +1 and -1 to the start and end of egfr series

def calc_gfr(scr, age, gender, ethnicity):
    """Calculates the estimated Glomerular Filteration Rate(eGFR)

    Based on MDRD equation.
    GFR=175×(Scr)^-1.154×(Age)^-0.203×(0.742 if female)×(1.212 if AA)

    Arguments:
        scr: Patient's serum creatinine level in mg/dL as float
        age: Patient's age in years as integer
        gender: Patient's gender "male" or "female" as string
        ethnicity: Patient's ethicity

    Returns:
        gfr: Patient's eGFR in mL/min rounded to decimal places.

    """
    if scr == 0.0 or age == 0.0:
        return 0

    if gender == 'F' and ethnicity == 'African American':
        egfr = round(175 * (scr) ** (-1.154) * age ** (-0.203) * (0.742) * 1.212, 2)
    elif gender == 'F' and ethnicity != 'African American':
        egfr = round(175 * (scr) ** (-1.154) * age ** (-0.203) * (0.742), 2)
    elif gender == 'M' and ethnicity == 'African American':
        egfr = round(175 * (scr) ** (-1.154) * age ** (-0.203) * 1.212, 2)
    else:
        egfr = round(175 * (scr) ** (-1.154) * age ** (-0.203), 2)
    return egfr




def get_AKI(egfr, nage):
    X = np.insert(np.diff(egfr), 0, 1)
    B = np.insert(X,len(X), -1)

    peak_indices = np.where(np.logical_and(B[:-1]<=0,B[1:]>=0))

    ln = np.insert(peak_indices, 0, 0)

    rate_of_change=np.array([])
    for i in range(0,len(ln)-1):
        temp = np.median(egfr[ln[i]:ln[i+1]])/egfr[ln[i + 1]]
        rate_of_change = np.append(rate_of_change,temp)

    idx = np.where(rate_of_change>1.5)
    AKI_idx  = np.array(idx)+1

    return{
        'no_of_AKI' : len(AKI_idx),
        'peak_indices': peak_indices,
        'loc' : ln,
        'egfr' : egfr,
        'AKI_idx' : AKI_idx,
        'nage' : nage
    }


if __name__ == '__main__':
    # example egfr and nage data
    #
    egfr = np.array([28.0486, 30.7319, 27.4494, 26.1951, 31.1516, 30.2168, 27.5738, 28.0995, 25.4419,
                     17.3429, 22.1788, 23.3516, 23.2128, 23.5936, 23.8530, 30.0470, 21.6360, 29.6093,
                     29.3810, 23.6897, 23.9733, 13.1291, 30.4058, 26.4273, 19.3072, 22.6676, 20.7426,
                     18.1169, 25.5715, 22.0056, 24.5697, 23.9159, 18.4622, 22.1596, 12.4535, 15.4332,
                     17.1968, 17.1968, 16.3719, 17.3497, 20.4957, 20.3742, 21.9593, 23.1170, 8.5815,
                     8.8424, 10.3382, 9.7079, 14.0047, 15.4073, 16.0679, 15.9973, 14.6092, 16.5627,
                     18.9653, 11.3656, 9.3886, 10.3613, 13.1368], dtype=np.float64)

    nage = np.array([64.4189, 64.7392, 64.8296, 65.0650, 65.1636, 65.4456, 65.5387, 66.0370, 66.2533,
                     66.3655, 66.4613, 66.4695, 66.5079, 66.9514, 67.2690, 67.2882, 67.3128, 67.3402,
                     67.5236, 67.5702, 67.5893, 68.2081, 68.2327, 68.5804, 68.6160, 68.6434, 68.7639,
                     68.7693, 68.7803, 69.0787, 69.5743, 70.4531, 70.4887, 70.5517, 70.7023, 70.7050,
                     70.7515, 70.7515, 70.8337, 70.8474, 71.0609, 71.2936, 71.7563, 71.9097, 72.4298,
                     72.5777, 72.5832, 72.5996, 72.7255, 72.7639, 72.8679, 72.9062, 72.9254, 72.9446,
                     73.1116, 73.7604, 73.7741, 73.9028, 74.3190], dtype=np.float64)

    result = get_AKI(egfr, nage)

    peak_indices = result['peak_indices']
    ln = result['loc']
    AKI_idx = result['AKI_idx']

    plt.plot(nage,egfr,'k-o', label = "egfr")
    # plt.plot(nage[peak_indices], egfr[peak_indices], 'r*')
    plt.tight_layout()

    plt.plot(nage[ln[AKI_idx]],egfr[ln[AKI_idx]],'rs',label = "AKI")
    plt.legend(loc='upper left', frameon=False)
    plt.show()