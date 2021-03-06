from hodgkinhuxleyfitter import HodgkinHuxleyFitter, HodgkinHuxleyFitterAdaptive, HodgkinHuxleyFitterFitfunFromSet
from channelfitter import ChannelFitterSingleTraces, ChannelFitterAllTraces


class FitterFactory:

    def __init__(self):
        pass

    def make_fitter(self, fitter_params):
        name = fitter_params['name']
        if name == 'HodgkinHuxleyFitter':
            return HodgkinHuxleyFitter(**fitter_params)
        elif name == 'HodgkinHuxleyFitterAdaptive':
            return HodgkinHuxleyFitterAdaptive(**fitter_params)
        elif name == 'HodgkinHuxleyFitterFitfunFromSet':
            return HodgkinHuxleyFitterFitfunFromSet(**fitter_params)
        # elif name == 'HodgkinHuxleyFitterCurrentPenalty':
        #     return  HodgkinHuxleyFitterCurrentPenalty(**fitter_params)
        # elif name == 'HodgkinHuxleyFitterPareto':
        #     return HodgkinHuxleyFitterPareto(**fitter_params)
        elif name == 'ChannelFitterSingleTraces':
            return ChannelFitterSingleTraces(**fitter_params)
        elif name == 'ChannelFitterAllTraces':
            return ChannelFitterAllTraces(**fitter_params)
        else:
            raise ValueError('Fitter '+name+' not available!')