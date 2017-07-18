from heka_reader import HekaReader
import os

if __name__ == '__main__':

    data_dir = '/home/cf/Phd/DAP-Project/cell_data/rawData'

    cells_by_protocol = dict()

    for file_name in os.listdir(data_dir):
        hekareader = HekaReader(os.path.join(data_dir, file_name))
        type_to_index = hekareader.get_type_to_index()

        group = 'Group1'
        protocol_to_series = hekareader.get_protocol(group)
        for protocol in protocol_to_series.keys():
            if protocol in cells_by_protocol:
                cells_by_protocol[protocol].append(file_name)
            else:
                cells_by_protocol[protocol] = [file_name]

    # search for protocol
    for protocol in cells_by_protocol:
        if 'zd' in protocol.lower():
            # show cells for protocol
            print cells_by_protocol[protocol]