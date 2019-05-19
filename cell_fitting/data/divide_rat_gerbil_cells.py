import pandas as pd


def check_rat_or_gerbil(cell_id):

    data_dir_gerbil = "/home/cfischer/Phd/DAP-Project/cell_data/division_rat_gerbil/Data+Immuno_Gerbil.csv"
    data_dir_rat = "/home/cfischer/Phd/DAP-Project/cell_data/division_rat_gerbil/Data+Immuno_Rat.csv"

    data_gerbil = pd.read_csv(data_dir_gerbil, header=1)
    data_rat = pd.read_csv(data_dir_rat, header=1)

    cells_gerbil = data_gerbil['Cellname'].apply(lambda x: x[1:-5]).values
    cells_rat = data_rat['Cellname'].apply(lambda x: x[1:-5]).values

    if cell_id in cells_gerbil:
        return 'gerbil'
    elif cell_id in cells_rat:
        return 'rat'
   # else:
        #print 'Neither in rat nor gerbil database!'
        #raise ValueError('Neither in rat nor gerbil database!')


if __name__ == '__main__':
    cell_id = '2015_03_30m'
    print check_rat_or_gerbil(cell_id)
