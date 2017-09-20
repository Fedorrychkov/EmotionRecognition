import numpy as np
from matplotlib import pyplot as plt


imsize1 = [300, 250, 200, 150, 100, 50]

score1 = [69.610091743119256, 71.788990825688074, 70.298165137614689, 69.724770642201833, 71.215596330275218, 64.105504587155963]
score2 = [71.330275229357795, 70.986238532110093, 71.559633027522935, 73.050458715596335, 72.477064220183493, 72.935779816513758]
score3 = [68.004587155963293, 65.022935779816521, 66.055045871559628, 64.564220183486242, 60.779816513761467, 48.394495412844037]

score11 = [17.201834862385319, 15.48165137614679, 16.169724770642201, 15.022935779816514, 15.481651376146788, 17.201834862385319]
score22 = [17.775229357798164, 16.743119266055047, 16.628440366972477, 18.11926605504587, 16.169724770642205, 16.284403669724771]
score33 = [15.711009174311927, 16.399082568807337, 18.11926605504587, 15.825688073394495, 15.25229357798165, 16.628440366972477]

plt.figure()
plt.plot(imsize1, score1, 'r')
plt.plot(imsize1, score11, 'g')
#plt.plot(imsize3, score3, 'y')
plt.show()
