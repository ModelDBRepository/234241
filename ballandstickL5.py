#!/usr/bin/env python

"""simple neuron model with soma and dendrite"""

# _title_     : neuron.py
# _author_     : Katharina Anna Wilmes
# _mail_     : katharina.anna.wilmes __at__ cms.hu-berlin.de


# --Imports--
import neuron
import matplotlib.pyplot as plt
import numpy as np
import nrn
import datetime
from neuron import h



# --load Neuron graphical user interface--
if not ( h('load_file("nrngui.hoc")')):
    print "Error, cannot open NEURON gui"



def make_compartment(length=150, diameter=1, nseg=5, Ra= 200, cm = 0.75):

    compartment = neuron.h.Section()
    compartment.L = length
    compartment.diam = diameter
    compartment.nseg = nseg
    compartment.Ra = Ra
    compartment.cm = cm
    return compartment


def fromtodistance(origin_segment, to_segment):
    h.distance(0, origin_segment.x, sec=origin_segment.sec)
    return h.distance(to_segment.x, sec=to_segment.sec)


class BiophysicsError(Exception):
    pass


class ShapeError(Exception):
    pass

    
class Neuron(object):
    """
    This class will produce Neuron objects with a soma and an apical dendrite
    """

    
    def __init__(self, params):

        # morphology
        self.a_diam = params['a_diam']
        self.s_diam = params['s_diam']        
        self.d_diam = params['d_diam']
        self.d_length = params['d_length']
        self.nseg = params['n_seg']

        
        # passive properties
        self.Rm = params['R_m'] 
        self.E = params['E_leak'] 
        self.Ra = params['R_a']
        self.cm = params['C_m']
        self.gp = 1/float(self.Rm)

        
        # active conductances
        self.ena = params['E_Na']
        self.ek = params['E_K']
        self.gna = params['g_Na']
        self.gk = params['g_K']
        self.gk_kap = params['g_KA']  
        self.slope = params['slope_KA']
        self.gna_ais = params['g_Na_ais']
        self.gna_ais_shifted = params['g_Na_ais_shifted']      
        
        # calcium
        self.eca = params['E_Ca']
        self.git2 = params['git2']
        self.gsca = params['gsca']
        self.gbar_kca = params['g_KCa']
        self.ifca = params['ifca']

        self.na_vshift = params['ifshift'] 
        self.ais_vshift = 10
        self.dend_vshift = params['dend_vshift']
  

        
        # creating compartments
        # axon inital segment
        self.ais = make_compartment(self.a_diam,self.a_diam*1.5,1)

        # soma
        self.soma = make_compartment(self.s_diam, self.s_diam,1)
        self.soma.connect(self.ais,1,0)
        self.dendrite = make_compartment(self.d_length,self.d_diam,self.nseg)
        self.dendrite.connect(self.soma,1,0)
        
        self.sinkdendrite = make_compartment(self.d_length,self.d_diam,self.nseg)
        
        self.sinkdendrite.connect(self.dendrite,1,0)
            
        #print(h.topology())

        # initialize parameters
        self.set_passive_parameters(self.gp, self.E, self.Ra)
        self.set_hh_parameters(self.ena, self.ek, self.gna, self.gk)

        self.set_kap_parameters(gkapbar = self.gk_kap, Ekap = self.ek)

        if self.ifca == True:
            self.set_ca_parameters(gsca = self.gsca, git2 = self.git2, 
            gkca = self.gbar_kca, eca = self.eca)

            
    def set_passive_parameters(self, gp, E, rho):
        for sec in neuron.h.allsec():
            sec.Ra = rho
            sec.insert("pas")
            for seg in sec:
                seg.pas.g = gp
                

    def set_hh_parameters(self, Ena, Ek, gnabar, gkbar):
        count = 0
        for sec in neuron.h.allsec():  
                count += 1
                sec.insert('na3dend')
                h.vshift_na3dend = self.dend_vshift
                sec.insert('kdr')
                for seg in sec:
                    seg.gbar_na3dend = gnabar 
                    seg.gkdrbar_kdr = gkbar 
                    seg.ena = Ena 
                    seg.ek = Ek 
                    
        if self.na_vshift:
            self.ais.gbar_na3dend = 0
            self.ais.insert('na3')
            self.ais.gbar_na3 = self.gna_ais
            self.ais.insert('na3shifted')
            h.vshift_na3shifted = self.ais_vshift
            self.ais.gbar_na3shifted = self.gna_ais_shifted
        elif not gnabar == 0:
            self.ais.gbar_na3 = self.gna_ais
        else: 
            print 'no Na in ais'
            

    def set_ca_parameters(self, gsca, git2, gkca, eca):
        h.distance(sec=self.soma, seg=0)
        for sec in neuron.h.allsec():
                sec.insert('sca')
                sec.insert('cad2')
                sec.insert('kca')
                for seg in sec: 
                    seg.eca = eca
                    #print seg.eca   
                    if not sec == self.soma:
                        sec.insert('it2')
                        dist = fromtodistance(self.soma(0.5),seg)
                        
                        if (dist > 500 and dist < 750):
                                seg.gcabar_it2 = git2 
                                seg.gbar_sca = gsca * 3 
                                seg.gbar_kca = gkca 
                        else:
                            seg.gcabar_it2 = 0
                            seg.gbar_sca = gsca 
                            seg.gbar_kca = gkca 

                    else:
                        for seg in sec:
                            seg.gbar_sca = gsca * 2 
                            seg.gbar_kca = gkca * 2 



    def set_kap_parameters(self, gkapbar, Ekap):
        h.distance(sec=self.soma)
        for sec in neuron.h.allsec():
            sec.insert('kap')
            if not sec == self.soma: 
                for seg in sec:
                    dist = fromtodistance(self.soma(0.5),seg)
                    if dist > 500:
                        dist = 500
                    seg.gkabar_kap = gkapbar*(1 + dist / (500 / self.slope))
                    seg.ek = Ekap
            else:
                self.soma.gkabar_kap = gkapbar
  