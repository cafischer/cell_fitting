from sklearn import linear_model
import numpy as np

save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/'
candidate_landscape_dirs = ['../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/v_trace',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/v_rest',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/APamp',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/APwidth',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/APtime',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/APshift',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/penalize_not1AP',
                            '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/phasehist']

optimum = np.array([0.12, 0.036])

p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p2_range.txt')

desired_landscape = np.zeros((len(p1_range), len(p2_range)))
for i, p1 in enumerate(p1_range):
    for j, p2 in enumerate(p2_range):
        desired_landscape[i, j] = (optimum - np.array([p1, p2])) ** 2
desired_landscape = desired_landscape.flatten()

candidate_landscapes = np.zeros((len(candidate_landscape_dirs), len(p1_range)*len(p2_range)))
for i, candidate_dir in enumerate(candidate_landscape_dirs):
    with open(save_dir+candidate_dir, 'r') as f:
        candidate_landscapes[i, :] = np.load(f).flatten()

regr = linear_model.LinearRegression()
regr.fit(candidate_landscapes, desired_landscape)


print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(candidate_landscapes) - desired_landscape) ** 2))
print('Variance score: %.2f'
      % regr.score(candidate_landscapes, desired_landscape))  # explained variance: 1 is perfect prediction