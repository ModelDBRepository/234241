
# coding: utf-8

# In[1]:

def set_axes(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_position(('outward', 10))
    axis.spines['left'].set_position(('outward', 10))
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')



# In[2]:

def get_viridis():
    _viridis_data = [[0.267004, 0.004874, 0.329415],
                 [0.268510, 0.009605, 0.335427],
                 [0.269944, 0.014625, 0.341379],
                 [0.271305, 0.019942, 0.347269],
                 [0.272594, 0.025563, 0.353093],
                 [0.273809, 0.031497, 0.358853],
                 [0.274952, 0.037752, 0.364543],
                 [0.276022, 0.044167, 0.370164],
                 [0.277018, 0.050344, 0.375715],
                 [0.277941, 0.056324, 0.381191],
                 [0.278791, 0.062145, 0.386592],
                 [0.279566, 0.067836, 0.391917],
                 [0.280267, 0.073417, 0.397163],
                 [0.280894, 0.078907, 0.402329],
                 [0.281446, 0.084320, 0.407414],
                 [0.281924, 0.089666, 0.412415],
                 [0.282327, 0.094955, 0.417331],
                 [0.282656, 0.100196, 0.422160],
                 [0.282910, 0.105393, 0.426902],
                 [0.283091, 0.110553, 0.431554],
                 [0.283197, 0.115680, 0.436115],
                 [0.283229, 0.120777, 0.440584],
                 [0.283187, 0.125848, 0.444960],
                 [0.283072, 0.130895, 0.449241],
                 [0.282884, 0.135920, 0.453427],
                 [0.282623, 0.140926, 0.457517],
                 [0.282290, 0.145912, 0.461510],
                 [0.281887, 0.150881, 0.465405],
                 [0.281412, 0.155834, 0.469201],
                 [0.280868, 0.160771, 0.472899],
                 [0.280255, 0.165693, 0.476498],
                 [0.279574, 0.170599, 0.479997],
                 [0.278826, 0.175490, 0.483397],
                 [0.278012, 0.180367, 0.486697],
                 [0.277134, 0.185228, 0.489898],
                 [0.276194, 0.190074, 0.493001],
                 [0.275191, 0.194905, 0.496005],
                 [0.274128, 0.199721, 0.498911],
                 [0.273006, 0.204520, 0.501721],
                 [0.271828, 0.209303, 0.504434],
                 [0.270595, 0.214069, 0.507052],
                 [0.269308, 0.218818, 0.509577],
                 [0.267968, 0.223549, 0.512008],
                 [0.266580, 0.228262, 0.514349],
                 [0.265145, 0.232956, 0.516599],
                 [0.263663, 0.237631, 0.518762],
                 [0.262138, 0.242286, 0.520837],
                 [0.260571, 0.246922, 0.522828],
                 [0.258965, 0.251537, 0.524736],
                 [0.257322, 0.256130, 0.526563],
                 [0.255645, 0.260703, 0.528312],
                 [0.253935, 0.265254, 0.529983],
                 [0.252194, 0.269783, 0.531579],
                 [0.250425, 0.274290, 0.533103],
                 [0.248629, 0.278775, 0.534556],
                 [0.246811, 0.283237, 0.535941],
                 [0.244972, 0.287675, 0.537260],
                 [0.243113, 0.292092, 0.538516],
                 [0.241237, 0.296485, 0.539709],
                 [0.239346, 0.300855, 0.540844],
                 [0.237441, 0.305202, 0.541921],
                 [0.235526, 0.309527, 0.542944],
                 [0.233603, 0.313828, 0.543914],
                 [0.231674, 0.318106, 0.544834],
                 [0.229739, 0.322361, 0.545706],
                 [0.227802, 0.326594, 0.546532],
                 [0.225863, 0.330805, 0.547314],
                 [0.223925, 0.334994, 0.548053],
                 [0.221989, 0.339161, 0.548752],
                 [0.220057, 0.343307, 0.549413],
                 [0.218130, 0.347432, 0.550038],
                 [0.216210, 0.351535, 0.550627],
                 [0.214298, 0.355619, 0.551184],
                 [0.212395, 0.359683, 0.551710],
                 [0.210503, 0.363727, 0.552206],
                 [0.208623, 0.367752, 0.552675],
                 [0.206756, 0.371758, 0.553117],
                 [0.204903, 0.375746, 0.553533],
                 [0.203063, 0.379716, 0.553925],
                 [0.201239, 0.383670, 0.554294],
                 [0.199430, 0.387607, 0.554642],
                 [0.197636, 0.391528, 0.554969],
                 [0.195860, 0.395433, 0.555276],
                 [0.194100, 0.399323, 0.555565],
                 [0.192357, 0.403199, 0.555836],
                 [0.190631, 0.407061, 0.556089],
                 [0.188923, 0.410910, 0.556326],
                 [0.187231, 0.414746, 0.556547],
                 [0.185556, 0.418570, 0.556753],
                 [0.183898, 0.422383, 0.556944],
                 [0.182256, 0.426184, 0.557120],
                 [0.180629, 0.429975, 0.557282],
                 [0.179019, 0.433756, 0.557430],
                 [0.177423, 0.437527, 0.557565],
                 [0.175841, 0.441290, 0.557685],
                 [0.174274, 0.445044, 0.557792],
                 [0.172719, 0.448791, 0.557885],
                 [0.171176, 0.452530, 0.557965],
                 [0.169646, 0.456262, 0.558030],
                 [0.168126, 0.459988, 0.558082],
                 [0.166617, 0.463708, 0.558119],
                 [0.165117, 0.467423, 0.558141],
                 [0.163625, 0.471133, 0.558148],
                 [0.162142, 0.474838, 0.558140],
                 [0.160665, 0.478540, 0.558115],
                 [0.159194, 0.482237, 0.558073],
                 [0.157729, 0.485932, 0.558013],
                 [0.156270, 0.489624, 0.557936],
                 [0.154815, 0.493313, 0.557840],
                 [0.153364, 0.497000, 0.557724],
                 [0.151918, 0.500685, 0.557587],
                 [0.150476, 0.504369, 0.557430],
                 [0.149039, 0.508051, 0.557250],
                 [0.147607, 0.511733, 0.557049],
                 [0.146180, 0.515413, 0.556823],
                 [0.144759, 0.519093, 0.556572],
                 [0.143343, 0.522773, 0.556295],
                 [0.141935, 0.526453, 0.555991],
                 [0.140536, 0.530132, 0.555659],
                 [0.139147, 0.533812, 0.555298],
                 [0.137770, 0.537492, 0.554906],
                 [0.136408, 0.541173, 0.554483],
                 [0.135066, 0.544853, 0.554029],
                 [0.133743, 0.548535, 0.553541],
                 [0.132444, 0.552216, 0.553018],
                 [0.131172, 0.555899, 0.552459],
                 [0.129933, 0.559582, 0.551864],
                 [0.128729, 0.563265, 0.551229],
                 [0.127568, 0.566949, 0.550556],
                 [0.126453, 0.570633, 0.549841],
                 [0.125394, 0.574318, 0.549086],
                 [0.124395, 0.578002, 0.548287],
                 [0.123463, 0.581687, 0.547445],
                 [0.122606, 0.585371, 0.546557],
                 [0.121831, 0.589055, 0.545623],
                 [0.121148, 0.592739, 0.544641],
                 [0.120565, 0.596422, 0.543611],
                 [0.120092, 0.600104, 0.542530],
                 [0.119738, 0.603785, 0.541400],
                 [0.119512, 0.607464, 0.540218],
                 [0.119423, 0.611141, 0.538982],
                 [0.119483, 0.614817, 0.537692],
                 [0.119699, 0.618490, 0.536347],
                 [0.120081, 0.622161, 0.534946],
                 [0.120638, 0.625828, 0.533488],
                 [0.121380, 0.629492, 0.531973],
                 [0.122312, 0.633153, 0.530398],
                 [0.123444, 0.636809, 0.528763],
                 [0.124780, 0.640461, 0.527068],
                 [0.126326, 0.644107, 0.525311],
                 [0.128087, 0.647749, 0.523491],
                 [0.130067, 0.651384, 0.521608],
                 [0.132268, 0.655014, 0.519661],
                 [0.134692, 0.658636, 0.517649],
                 [0.137339, 0.662252, 0.515571],
                 [0.140210, 0.665859, 0.513427],
                 [0.143303, 0.669459, 0.511215],
                 [0.146616, 0.673050, 0.508936],
                 [0.150148, 0.676631, 0.506589],
                 [0.153894, 0.680203, 0.504172],
                 [0.157851, 0.683765, 0.501686],
                 [0.162016, 0.687316, 0.499129],
                 [0.166383, 0.690856, 0.496502],
                 [0.170948, 0.694384, 0.493803],
                 [0.175707, 0.697900, 0.491033],
                 [0.180653, 0.701402, 0.488189],
                 [0.185783, 0.704891, 0.485273],
                 [0.191090, 0.708366, 0.482284],
                 [0.196571, 0.711827, 0.479221],
                 [0.202219, 0.715272, 0.476084],
                 [0.208030, 0.718701, 0.472873],
                 [0.214000, 0.722114, 0.469588],
                 [0.220124, 0.725509, 0.466226],
                 [0.226397, 0.728888, 0.462789],
                 [0.232815, 0.732247, 0.459277],
                 [0.239374, 0.735588, 0.455688],
                 [0.246070, 0.738910, 0.452024],
                 [0.252899, 0.742211, 0.448284],
                 [0.259857, 0.745492, 0.444467],
                 [0.266941, 0.748751, 0.440573],
                 [0.274149, 0.751988, 0.436601],
                 [0.281477, 0.755203, 0.432552],
                 [0.288921, 0.758394, 0.428426],
                 [0.296479, 0.761561, 0.424223],
                 [0.304148, 0.764704, 0.419943],
                 [0.311925, 0.767822, 0.415586],
                 [0.319809, 0.770914, 0.411152],
                 [0.327796, 0.773980, 0.406640],
                 [0.335885, 0.777018, 0.402049],
                 [0.344074, 0.780029, 0.397381],
                 [0.352360, 0.783011, 0.392636],
                 [0.360741, 0.785964, 0.387814],
                 [0.369214, 0.788888, 0.382914],
                 [0.377779, 0.791781, 0.377939],
                 [0.386433, 0.794644, 0.372886],
                 [0.395174, 0.797475, 0.367757],
                 [0.404001, 0.800275, 0.362552],
                 [0.412913, 0.803041, 0.357269],
                 [0.421908, 0.805774, 0.351910],
                 [0.430983, 0.808473, 0.346476],
                 [0.440137, 0.811138, 0.340967],
                 [0.449368, 0.813768, 0.335384],
                 [0.458674, 0.816363, 0.329727],
                 [0.468053, 0.818921, 0.323998],
                 [0.477504, 0.821444, 0.318195],
                 [0.487026, 0.823929, 0.312321],
                 [0.496615, 0.826376, 0.306377],
                 [0.506271, 0.828786, 0.300362],
                 [0.515992, 0.831158, 0.294279],
                 [0.525776, 0.833491, 0.288127],
                 [0.535621, 0.835785, 0.281908],
                 [0.545524, 0.838039, 0.275626],
                 [0.555484, 0.840254, 0.269281],
                 [0.565498, 0.842430, 0.262877],
                 [0.575563, 0.844566, 0.256415],
                 [0.585678, 0.846661, 0.249897],
                 [0.595839, 0.848717, 0.243329],
                 [0.606045, 0.850733, 0.236712],
                 [0.616293, 0.852709, 0.230052],
                 [0.626579, 0.854645, 0.223353],
                 [0.636902, 0.856542, 0.216620],
                 [0.647257, 0.858400, 0.209861],
                 [0.657642, 0.860219, 0.203082],
                 [0.668054, 0.861999, 0.196293],
                 [0.678489, 0.863742, 0.189503],
                 [0.688944, 0.865448, 0.182725],
                 [0.699415, 0.867117, 0.175971],
                 [0.709898, 0.868751, 0.169257],
                 [0.720391, 0.870350, 0.162603],
                 [0.730889, 0.871916, 0.156029],
                 [0.741388, 0.873449, 0.149561],
                 [0.751884, 0.874951, 0.143228],
                 [0.762373, 0.876424, 0.137064],
                 [0.772852, 0.877868, 0.131109],
                 [0.783315, 0.879285, 0.125405],
                 [0.793760, 0.880678, 0.120005],
                 [0.804182, 0.882046, 0.114965],
                 [0.814576, 0.883393, 0.110347],
                 [0.824940, 0.884720, 0.106217],
                 [0.835270, 0.886029, 0.102646],
                 [0.845561, 0.887322, 0.099702],
                 [0.855810, 0.888601, 0.097452],
                 [0.866013, 0.889868, 0.095953],
                 [0.876168, 0.891125, 0.095250],
                 [0.886271, 0.892374, 0.095374],
                 [0.896320, 0.893616, 0.096335],
                 [0.906311, 0.894855, 0.098125],
                 [0.916242, 0.896091, 0.100717],
                 [0.926106, 0.897330, 0.104071],
                 [0.935904, 0.898570, 0.108131],
                 [0.945636, 0.899815, 0.112838],
                 [0.955300, 0.901065, 0.118128],
                 [0.964894, 0.902323, 0.123941],
                 [0.974417, 0.903590, 0.130215],
                 [0.983868, 0.904867, 0.136897],
                 [0.993248, 0.906157, 0.143936]]

    from matplotlib.colors import ListedColormap

    cmaps = {}
    cmaps['viridis'] = ListedColormap(_viridis_data, name='viridis')
    cmaps['viridis_r'] = ListedColormap(_viridis_data[::-1], name='viridis_r')


    viridis = cmaps['viridis']
    viridis_r = cmaps['viridis_r']
    cm.register_cmap(cmap=viridis)
    cm.register_cmap(cmap=viridis_r)
    


# In[3]:

# some functions

def truncated_gauss(N, mu, sigma=0.05, a=0, b=2): 
    """Return a N random numbers from a truncated (a,b) Gaussian distribution.""" 

    pos = np.zeros(N)
    i = 0
    while i<N: 
        x = gauss(mu, sigma)
        #x = np.random.normal(mu,sigma) 
        if a <= x <= b: 
            pos[i] = round(x,2)
            i += 1
    return pos

def get_uniform(N, a=0, b=1):
    rs_spatial = RandomState(params['Inhibition']['pos_seed'])
    pos = (rs_spatial.rand(N)*a)+(b-a) # a=0.3 and b =0.6 to get values between 0.3 and 0.6, which is between 100 and 300 um
    return pos



def get_interneuron():
    interneuron = neuron.h.Section()
    interneuron.L = 67
    interneuron.diam = 67 # so that area is about 14000 um2
    interneuron.nseg = 1
    interneuron.Ra = 100
    interneuron.cm = 1
    
    interneuron.insert('pas')
    for seg in interneuron:
        seg.pas.g = 0.00015
        seg.pas.e = -70
    
    interneuron.insert('hh2')
    interneuron.vtraub_hh2 = -55 #resting Vm, BJ was -55
    interneuron.gnabar_hh2 = 0.05 #McCormick=15 muS, thal was 0.09
    interneuron.gkbar_hh2 = 0.01 #spike duration of interneurons
    interneuron.ena = 50 
    interneuron.ek = -100

    return interneuron
    
def get_netstim(no_reps,freq,kind):
    AP_DELAY  = 7.5 # approximate time between EPSP onset and AP peak
    ns = h.NetStim()
    ns.interval = 1000.0/freq
    ns.start = WARM_UP-AP_DELAY # ms (most likely) start time of first spike  
    
    if kind == "Poisson":
        ns.number = 1e9 # (average) number of spikes
        ns.noise = 1 
    elif kind == "deterministic":
        ns.number = no_reps # (average) number of spikes
    else:
        print "kind of input unknown"
        return ValueError
        
    return ns


# In[4]:

import neuron
from neuron import h
import sys
from ballandstickL5 import *
from numpy.random import RandomState
from random import gauss
import matplotlib.cm     as cm


# Parameter Settings

# In[5]:

params = {
    'visual': 'figure',
    'results_file': 'pairingscenarios',
    'Input':
        {
        'freq': 10,
        'kind': 'deterministic'

        },
    'Neuron':
        {
        # morphology
        'a_diam':2,
        's_diam':18.5,
        'd_diam':2,
        'd_length':500,              
        'n_seg':201,
        # passive parameters
        'R_m':40000,
        'R_a':150, 
        'C_m':0.75,              
        'E_leak':-70,
        'V_rest':-70,
        # active conductances
        'E_Na':60,
        'E_K':-80,
        'E_Ca':140,
        'g_Na':0.009,
        'g_K':0.01,
        'g_KA':0.029, 
        'slope_KA':5,
        ##calcium
        'gsca': 1.5, 
        'git2': 0.009, 
        'g_KCa':2.5, 
        'ifca': False,
        # ais
        'g_Na_ais':0.3,
        'g_Na_ais_shifted':0.3,
        'ifshift':True,
        'dend_vshift':5
        },
    'Stimulation':
        {
        'amp':0.3        
        },
    'Excitation':
        {
        'freq': 10,
        'w_ee': 0.005, 
        'w_ei': 0.2,
        'no_epsps': 8,
        'tau1': 0.5,
        'tau2': 2
        },
    'Inhibition':
        {
        'shunt_reversal':-74,
        'pos': 0.4,
        'tau1':0.5,
        'tau2':5,
        'delay':0,
        'weight':0.00001, 
        'timing' : -1, 
        'random': 'random',
        'pos_seed': 6223905,        
        'seed': 4260404,
        'jitter_sigma': 0.5,
        'pos_sigma': 0, #0.02
        'p_inh':1,
        'weight_distribution':'Delta', #'Exponential', 'Beta'
        'spatial_distribution': 'Normal'#'Normal'
        },
    
   'STDP':
       {
        'delta_t': 0, 
        'thresh' : -40,
        'shift'  : 1.27, 
        'potentiation': 0.01,
        'depression'  : 0.01,
        'tau_p' : 10,
        'tau_d' : 10,
        'wmax': 0.01,
        'alpha': 0.005,
        'rule': 'anti-Hebbian', 
        'learning_rate': 1, 
        'no_reps': 100,
        'pot_l':0.02,
        'pot_r':1.27
       },    
    'sim':
        {
        'duration' : 1,
        },
    'plot':
        {
        'version':1        
        }
}
     


# In[6]:

NO_INH = 100 # number of inhibitory synapses
NO_REPS = 100 #number of pairings
DT=0.025 # ms, integration time step
POST_AMP = 0.2 # nA, amplitude of current injection to trigger the POST-synaptic spike
WARM_UP=100 # ms

freq = params['Input']['freq']
kind = params['Input']['kind']

delta_t = params['STDP']['delta_t']
learning_rate = params['STDP']['learning_rate']
no_reps = params['STDP']['no_reps']

weight_distribution = params['Inhibition']['weight_distribution']
inh_pos = params['Inhibition']['pos']
pos_sigma = params['Inhibition']['pos_sigma']

sigma = params['Inhibition']['jitter_sigma']


# Circuit elements

# In[7]:

# create cell
cell = Neuron(params['Neuron'])

# create interneuron    
interneuron = get_interneuron()    

# excitatory input neuron      
ns = get_netstim(no_reps,freq, kind)


# Connections

# In[8]:

# excitatory synapse
w_ee = params['Excitation']['w_ee']
no_epsps = params['Excitation']['no_epsps']
ex_ex = h.List()  
nc_ex = h.List()
for i in range(1,no_epsps+1):                    
    pos = 0.3 + i * 0.075            
    ex = neuron.h.Exp2Syn(cell.dendrite(pos))
    w = w_ee/no_epsps #0.016
    ex.tau1 = params['Excitation']['tau1']  # ms rise time
    ex.tau2 = params['Excitation']['tau2']  # ms decay time
    ex.e = 0 # mV reversal p
    # excite postsynaptic cell
    nc = h.NetCon(ns,ex,1,0,w)    
    ex_ex.append(ex)
    nc_ex.append(nc)


# excitatory to inhibitory synapse        
ex_inh = neuron.h.Exp2Syn(interneuron(0.5))
ex_inh.tau1 = 0.5 # ms rise time
ex_inh.tau2 = 2 # ms decay time
ex_inh.e = 0 # mV reversal p

# excite interneuron
w_ei = params['Excitation']['w_ei']
nc_ex_inh = h.NetCon(ns,ex_inh,1,0,w_ei)

total_time = WARM_UP+no_reps*(1000.0/freq)+100

# plastic inhibitory synapses
exs = h.List()
exnc = h.List()
syns = h.List()

w = []

# weights of inhibitory synapses
w_ie = params['Inhibition']['weight']
if weight_distribution == 'Delta':
    inh_weights = w_ie * np.ones(NO_INH)
elif weight_distribution == 'Exponential':
    inh_weights = w_ie * np.random.exponential(size = NO_INH)
elif weight_distribution == 'Beta':
    inh_weights = 2 * w_ie * np.random.beta(0.5,0.5,size = NO_INH)
elif weight_distribution == 'Normal':
    inh_weights = np.random.normal(w_ie,w_ie/2,size = NO_INH)
else:
    raise NameError('Option %s does not exist'%weight_distribution)

# timing of inhibitory synapses 
timing = np.zeros((NO_INH))
inh_delay = np.zeros((NO_INH))
if params['Inhibition']['random'] == 'random':
    rs = RandomState(params['Inhibition']['seed'])
    timing = (rs.rand(NO_INH)*10)+1
    timing[timing<0] == 0
elif params['Inhibition']['random'] == 'limited':
    timing = np.linspace(1,5.0,num=NO_INH)
else:
    timing = np.linspace(1,11,num=NO_INH)    

# position of inhibitory synapses
if params['Inhibition']['spatial_distribution'] == 'Normal':        
    inh_poses = truncated_gauss(NO_INH, inh_pos, pos_sigma, a=0, b=1) 
elif params['Inhibition']['spatial_distribution'] == 'Uniform':
    inh_poses = get_uniform(NO_INH, a=inh_pos-0.1, b=inh_pos+0.1) 
else:
    raise ValueError
inh_poses.sort()

# set inhibitory synapses
for inh in np.arange(NO_INH):
    inh_delay[inh] = timing[inh]-(sigma*3)
    if inh_delay[inh]<0:
        inh_delay[inh]=0
    if not weight_distribution == 'Delta':
        if inh_weights[inh]<=0:
            inh_weights[inh]=0.0000000001
        w_ie = inh_weights[inh]

    shift = params['STDP']['shift']
    if params['STDP']['rule'] == 'mexican':
        if shift >= 0:
            syn = h.Mexhat_Inh_STDP(cell.dendrite(inh_poses[inh]))
            syn.shift = shift
            syn.alpha = params['STDP']['mex_alpha']
        else:
            syn = h.Mexhat_Inh_STDP_n(cell.dendrite(inh_poses[inh]))
            syn.shift = -shift
            syn.alpha = params['STDP']['mex_alpha']
    elif params['STDP']['rule'] == 'anti-Hebbian':
        if shift >= 0:
            syn = h.Exp2Syn_Inh_STDP_nobound(cell.dendrite(inh_poses[inh])) #opt2 without bounds
            syn.shift = shift
            syn.learning_rate = learning_rate
        else:
            syn = h.Exp2Syn_Inh_STDP_n(cell.dendrite(inh_poses[inh]))    
            syn.shift = -shift
    elif params['STDP']['rule'] == 'cut':
            syn = h.Exp2Syn_Inh_STDP_cut_nobound(cell.dendrite(inh_poses[inh])) #opt2 without bounds
            syn.shift = shift
            syn.cut = params['STDP']['pot_l']
            syn.learning_rate = learning_rate
    elif params['STDP']['rule'] == 'optimal':
            syn = h.Exp2Syn_Inh_STDP_opt2(cell.dendrite(inh_poses[inh])) #opt2 without bounds
            syn.pot_l = params['STDP']['pot_l']
            syn.pot_r = params['STDP']['pot_r']
            syn.learning_rate = learning_rate
    elif params['STDP']['rule'] == 'cut_twice':
            syn = h.Exp2Syn_Inh_STDP_cuttwice(cell.dendrite(inh_poses[inh])) #opt2 without bounds
            syn.shift = shift
            syn.cut = params['STDP']['pot_l']
            syn.learning_rate = learning_rate

    else:
        raise ValueError

    syn.tau1 = params['Inhibition']['tau1']
    syn.tau2 = params['Inhibition']['tau2']
    syn.e = params['Inhibition']['shunt_reversal']
    syn.thresh = params['STDP']['thresh']        
    syn.dd = params['STDP']['potentiation']
    syn.dp = params['STDP']['depression']
    syn.ptau = params['STDP']['tau_p']
    syn.dtau = params['STDP']['tau_p']
    syn.wmax = params['STDP']['wmax']
    syn.mean = sigma*3
    syn.std = sigma
    syns.append(syn)   
    # inh_delay may not be negative
    exnc.append(h.NetCon(interneuron(0.5)._ref_v, syn,0,inh_delay[inh],w_ie, sec = interneuron))
    tvec = h.Vector()
    exnc[inh].record(tvec)
    wrec = h.Vector()
    wrec.record(exnc[inh]._ref_weight[3])
    w.append(wrec)

syn_cond = h.List()
for i in np.arange(len(syns)):
    syn_cond.append(neuron.h.Vector())
    syn_cond[i].record(syns[i]._ref_g)


# In[9]:

# recording

trec = h.Vector()
trec.record(h._ref_t)
rec_v = neuron.h.Vector()
rec_v1 = neuron.h.Vector()
rec_v2 = neuron.h.Vector()
rec_v3 = neuron.h.Vector()
rec_v4 = neuron.h.Vector()
rec_v5 = neuron.h.Vector()
rec_v6 = neuron.h.Vector()
rec_v7 = neuron.h.Vector()
rec_v8 = neuron.h.Vector()
rec_v9 = neuron.h.Vector()
rec_v.record(cell.soma(0.5)._ref_v)
rec_v1.record(cell.dendrite(0.1)._ref_v)
rec_v2.record(cell.dendrite(0.2)._ref_v)
rec_v3.record(cell.dendrite(0.3)._ref_v)
rec_v4.record(cell.dendrite(0.4)._ref_v)
rec_v5.record(cell.dendrite(0.5)._ref_v)
rec_v6.record(cell.dendrite(0.6)._ref_v)
rec_v7.record(cell.dendrite(0.7)._ref_v)
rec_v8.record(cell.dendrite(0.8)._ref_v)
rec_v9.record(cell.dendrite(0.9)._ref_v)

vinhrec = h.Vector()
vinhrec.record(interneuron(0.5)._ref_v)   

grec = h.Vector()
grec.record(exnc[0]._ref_weight[1])



# Run Simulation

# In[10]:

h.dt = DT
h.celsius = 30
h.finitialize(-70)
neuron.run(total_time)


# In[11]:

# data collection
sampling_start = WARM_UP+50
sampling_interval = 1000.0/freq
t = np.array(trec)
inh_spikes = np.array(tvec)
v = np.array(rec_v)
v_inh = np.array(rec_v4)
vinh = np.array(vinhrec)
vd = np.array(rec_v9)
g = np.array(grec)
w = np.array(w)

my_rawdata = {}
my_rawdata['timing'] = timing    
my_rawdata['v'] = v    
my_rawdata['vd'] = vd
sampling_start = int((WARM_UP+50)/DT)
sampling_interval = int((1000.0/freq)/DT)
my_rawdata['w'] = w   
my_rawdata['t'] = t 
my_rawdata['v_inh'] = v_inh 
my_rawdata['vinh'] = vinh 
my_rawdata['inh_delay'] = inh_delay 
my_rawdata['inh_spikes'] = inh_spikes 
my_rawdata['inh_poses'] = inh_poses 
my_rawdata['weight_distribution'] = inh_weights
rawdata = {'raw_data':my_rawdata}


# Visualization

# In[12]:

get_viridis()
mycmap = cm.get_cmap('viridis') 
interval = 1000/freq
AP_amp = np.zeros((NO_REPS))
bAP_amp = np.zeros((NO_REPS))
bAP_distal_amp = np.zeros((NO_REPS))

j=0
for i in np.arange(NO_REPS):
    start = int(i*interval/DT+interval/DT-100)
    end = int(i*interval/DT+interval/DT+100)      
    bAP = vd[start:end]
    AP_amp[j] = np.max(v[start:end])
    AP_time = np.argmax(v[start:end])
    bAP_amp[j] = np.max(v_inh[start:end])
    bAP_distal_amp[j] = np.max(vd[start:end])
    j+=1


# Figure 2B top

# In[13]:

ax = plt.figure()
j=1
for i in [0,NO_REPS-1]:
    axis = plt.subplot(1,2,j)
    j+=1
    plt.plot(t,v,'k',lw=3 ,label = 'AP')
    plt.plot(t,vd,color= mycmap(0.4),lw=3, linestyle='dashed', label = 'bAP')

    xmin = (i * interval)+ interval-10
    xmax = (i * interval)+ interval+10#100
    plt.xlim((xmin,xmax))
    plt.ylim((-80,60))

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    plt.xlabel("p %d"%(i+1))
    if i==0:        
        axis.spines['bottom'].set_visible(False)            
        axis.spines['left'].set_position(('outward', 10))
        axis.yaxis.set_ticks_position('left')
        plt.xticks([])            
        plt.ylabel("membrane potential [mV]")
    else:
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        axis.legend(prop={'size':15}, frameon = False)
plt.show()


# Figure 2B and D

# In[14]:

AP_time = (np.argmax(v[20*interval*int(1/DT)+100:21*interval*int(1/DT)+100])+100)*DT
bAP_thresh_crossing1= (v_inh[20*interval*int(1/DT)+100:21*interval*int(1/DT)+100]>-40)
bAP_thresh_crossing1= (v_inh[0*interval*int(1/DT)+100:1*interval*int(1/DT)+100]>-40)
bAP_thresh_crossing2 = np.nonzero(bAP_thresh_crossing1)
try:    
    bAP_thresh_crossing = (bAP_thresh_crossing2[0][0]+100)*DT
    diff = bAP_thresh_crossing - AP_time
except IndexError:
    pass
inh_times = inh_spikes[0]+timing   

try:
    ref_time = bAP_thresh_crossing
except IndexError:    
    ref_time = AP_time    
timing_relative = ref_time-inh_times
timing_range = np.max(timing_relative)-np.min(timing_relative)



# In[15]:

sm3 = plt.cm.ScalarMappable(cmap='viridis_r')
sm3.set_array(timing_relative)    

fig = plt.figure(figsize = (9,13.5),dpi=80)
ax = plt.subplot(211)
set_axes(ax)

ax.plot(np.arange(len(AP_amp[:])),AP_amp[:],'k-',lw=3, label = 'somatic AP')
ax.plot(np.arange(len(bAP_amp[:])),bAP_amp[:],color = mycmap(0.8), linestyle = 'dotted', lw=3, label = 'bAP at inh')
ax.plot(np.arange(len(bAP_distal_amp[:])),bAP_distal_amp[:],color=mycmap(0.4), linestyle = 'dashed', lw=3, label = 'distal bAP')
plt.xlabel("number of pairings",fontsize = 'xx-large')
plt.ylabel("membrane potential [mV]",fontsize = 'xx-large')
ax.legend(prop={'size':15}, frameon = False)     
plt.xticks(np.arange(0,NO_REPS+1,20),np.arange(0,NO_REPS+1,20))        

pairingrange = np.arange(len(bAP_distal_amp[:]))    
bAP_fail = pairingrange[bAP_distal_amp[:]<-20]
bAP_fail = np.min(bAP_fail)
plt.title("bAP fails at pairing %d"%bAP_fail)
plt.ylim(-80,40)
      

ax2 = plt.subplot(212)
set_axes(ax2)

for i in np.arange(len(w[:,0])):
    c = ((-timing_relative[i]+abs(np.min(timing_relative))))/timing_range
    ax2.plot(np.arange(len(w[i,:NO_REPS*interval*int(1/DT)+1])),((w[i,:NO_REPS*interval*int(1/DT)+1]-w[i,0])/w[i,0]),color = mycmap(c))
    plt.xticks(np.arange(0,NO_REPS*interval*int(1/DT)+1,interval*int(1/DT)*20),np.arange(0,NO_REPS+1,20))        
    plt.ylim(-10,1000)
    ax2.set_yscale("symlog", linthreshx=1)
    plt.xlabel("number of pairings",fontsize = 'xx-large')
    plt.ylabel("synaptic weight [nS]",fontsize = 'xx-large')
cb = fig.colorbar(sm3)
cb.set_label('relative timing $[ms]$', fontsize = 28)
plt.show() 


# Figure 2C

# In[16]:

plt.figure()
count = 1
ax = plt.subplot(111)
set_axes(ax)
w_fix = w[:,-1]
w_fix = w_fix*1000
w_fix1 = w_fix[timing_relative<1.27] 
w_fix2 = w_fix[timing_relative>=1.27]
w_fix1_wrong = w_fix1[w_fix1<1]
bins=np.histogram(np.hstack((w_fix1,w_fix2)), bins=40)[1] #get the bin edges
p1 = ax.hist(w_fix1, bins, color=mycmap(0.8), edgecolor = 'None', label = 'Delta t < 1.27ms')
p2 = ax.hist(w_fix2, bins, color=mycmap(0), edgecolor = 'None', label = 'Delta t >= 1.27ms')
p1 = ax.hist(w_fix1_wrong, bins, color=mycmap(0.8), edgecolor = 'None')

plt.xlabel("synaptic weight [nS]", fontsize = 'xx-large')
ax.legend(prop={'size':15}, frameon = False)
plt.tight_layout() 
plt.show()


# In[ ]:




# In[ ]:



