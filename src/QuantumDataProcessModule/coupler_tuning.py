import qdpm
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
exp = qdpm.Experiment(r'O:\68707\BF2222\LABVIEWdata\Cooldown20200722\Heterodyne\PTC\Coupler tuning\2020_07_24_19_33_28')
print(exp.scan_value[0][80])
fig, ax = plt.subplots()
ax.pcolormesh(exp.mean_mag()[0, :, :].T)
ax.set_xticks(np.arange(0, 201, 20))
ax.set_yticks(np.arange(0, 21, 5))
ax.set_xticklabels(exp.scan_value[0][0:201:20])
ax.set_yticklabels(exp.scan_value[1][0:21:5])
plt.show()

