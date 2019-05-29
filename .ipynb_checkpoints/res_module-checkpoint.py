import os
import numpy as np
import gzip
import timeit
import MDAnalysis as mda
import warnings
import datetime

warnings.filterwarnings("ignore")


def remove_close_id(dict_id, dict_dt, thr_clos_id):
    """
        Function which takes as input a dictionary of collected residues id 
        and a dictionary of corresponding distances removes those ids 
        which are closer then thr_clos_id.
        
        Inputs:
            dict_id: dictionary of residues ids
            dict_dt: dictionary of residues dts
            thr_clos_id: threshold of number of ids of distance 
            
        Returns:
            dictionary of filtered ids
    """
    _aux = {key: value[:] for key, value in dict_id.items()}   # Deep copy of a dictionary
    for key in _aux:                                           # Iterate through the dictionary 
        if len(_aux[key]) < 3:
            dict_id.pop(key)
        else:
            n = len(_aux[key])
            for i in range(n - 1):
                if np.abs(_aux[key][i] - _aux[key][i + 1]) <= thr_clos_id:
                    if _aux[key][i]     in dict_id[key]: dict_id[key].remove(_aux[key][i])
                    if _aux[key][i + 1] in dict_id[key]: dict_id[key].remove(_aux[key][i + 1])
    return dict_id


def eval_centroid(dict_res):
        """
        Function which takes as input a dictionary of residues 
        and evaluates the centroid of each set of coordinates.
        
        Inputs:
            dict_res: dictionary of residues:positions
            
        Returns:
            numpy array of centroids
        """
        return np.array([np.sum(np.array(dict_res[i])[:,0])/np.shape(np.array(dict_res[i]))[0] for i in dict_res.keys()]) 

def build_residues_dict(positions, resid_ids, global_centroid):
    """
        Function which takes as input positions (coordinates), residues ids and 
        the protein centroid extracted as MDAnalysis and build a dictionary d{resid:[coordinates]}
        
        Inputs:
            positions : residue.position
            resid_ids : residue.resid 
            global_centroid : all_atom.centroid()
        
        Returns:
            d{resid:[coordinates]} dictionary
    """
    d = {}
    for res in zip(positions, resid_ids):
        if res[1] in d:
            d[res[1]].append(res[0] )
        else:
            d[res[1]] = []
            d[res[1]].append(res[0] )
    return d


def build_dist_res(d_id,d_dt):
    """
        Function which takes as input the dictionary of the residues ids
        and the dictionary of the residues distances and build a new 
        dictionary organize as:
        d['id1-id2'] = distance(id1-id2)
        
        Inputs:
            d_id : dictionary of residues id
            d_dt : dictionary of residues distances
        
        Returns:
            dictionary of combined distances        
    """
    count = 0
    c_dt  = {}
    for i in d_id:
        count = 0
        for j in d_id[i]:
            c_dt[str(i) + '-' + str(j)] = d_dt[i][count]
            count +=1
    return c_dt

def eval_distance(centroid_1, centroid_2):
    """
        Evaluate the distance between the centroid of the first and the 
        second residue 
        
        Inputs:
            centroid_1: numpy array of centroid coordinates for each residue 1 of len n
            centroid_2: numpy array of centroid coordinates for each residue 2 of len m
        
        Returns:
             multidimensional numpy array [n x m] of distance between centroid 1 and centroid 2
    """
    return np.round(np.array([[np.abs(i - j) for i in centroid_1] for j in centroid_2]),3)

def order_residue_dict(bool_dis, dis, keys_1, keys_2 ):
    """
        Function which takes as input a boolean distance matrix bool_dis,
        the corresponding distance matrix dis and build two dictionary
        d_id {resid_1:[close_resid_2]} 
        d_dt {resid_1:[close_dist_2]}
        where resid 1 are the residues ids in keys_1 and resid_2 are the 
        residues ids in keys_2
        
        Inputs:
            bool_dis : boolean distance matrix
            dis : distance matrix
            keys_1 : residues ids 1 --> keys of the new dictionary
            keys_2 : residues ids 2 --> related values of the new dictionary
        
        Returns:
            d_id{ resid_1: close_resid_2}    dictionary of which residue ids 2 are close to the residue 1
            d_dt{ resid_1: close_distace_2} dictionary of the distance value of the residue ids 2 that are close to the residue 1
    """
    
    d_id, d_dt = {}, {}
    for i, resid in enumerate(keys_1):
        close_resid    = list(np.multiply(bool_dis[i], keys_2))
        close_dist     = list(np.round(np.multiply(bool_dis[i], dis[i]),3))
        
        for i,v in enumerate(close_resid):
            if resid not in d_id:
                d_id[resid], d_dt[resid] = [], []
                if v != 0:
                    d_id[resid].append(close_resid[i])
                    d_dt[resid].append(close_dist[i])
            else:
                if v != 0:
                    d_id[resid].append(close_resid[i])
                    d_dt[resid].append(close_dist[i])
    return d_id, d_dt

def score_function(cta_id, ctg_id, a_dt):
    """
        Function which takes in input two dictionary, (id closer res1-res2, id closer res2-res2)
        and count the occurence of a res 2 close to res1 in the residue near res2
        
        Inputs:
            cta_id dictionary cta_id[id-res1] = [list of id-res2 closer]
            ctg_id dictionary cta_id[id-res2] = [list of id-res2 closer]
            a_dt   dictionary   a_dt[id1-id2] = distance(id1-id2)
        
        Returns:
            score_tot  : a nested dictionary of the number of occurence of a res 2 to be close 
                to a res1 and another res2: score_tot[id-res1] = score[id-res2]:occurence
            score_list : a dictionary of the res 1 close to res 2 after the score evaluation
            score_dt   : a dictionary of the distances between res 1 and res 2 after the score evaluation
            
    """
    score, score_id, score_dt, score_list = {}, {}, {}, {}
    for a_key in cta_id:
        # Iterate through the ASP
        if len(cta_id[a_key]) > 2:
            # If more then 2 GLU are close
            for g_key in cta_id[a_key]:
                # Iterate through the GLU closer
                if g_key in ctg_id:
                    for res in ctg_id[g_key]:
                        if res in cta_id[a_key]:
                            if res in score: score[res] +=1
                            else:
                                score[res] = 1
        _aux = {key: value for key, value in score.items() if value !=1} # remove value if has just one occurence
        if _aux:
            score_id[a_key]   = _aux
            score_list[a_key] = [i for i in _aux]
            score_dt[a_key]   = np.array([a_dt[str(a_key) + '-' + str(i)] for i in _aux])
        score, _aux = {}, {}
    return score_id, score_list, score_dt


def search_quad_procedure(U, initial_params):
    """
        Function which takes as input an universe created
        by the MDAnalysis tool, a set of initial parameters
        for the search and look for a set of residues in the
        universe in a quadrilateral conformation as in the case
        of the yeast frataxin residues GLU:89 - GLU:112 - ASP:101 - GLU:103
        
        Inputs:
            U : universe created with the MDAnalysis tool of a pdb file
            initial params : set of the initial parameters for the search
        
        Returns:
            score : set of residues which are similar to the one 
                    described in the yeast frataxin 
    """
    # Initialization parameters for the search
    am_1        = initial_params['aminoacid_1']
    am_2        = initial_params['aminoacid_2']
    thr_glu_glu = initial_params['thr_glu_glu']
    thr_glu_asp = initial_params['thr_glu_asp']
    thr_clos_id = initial_params['thr_clos_id']
    
    # Get atoms of all the aminoacid_1 and aminoacid_2
    all_atom     = U.atoms
    centroid_at  = all_atom.centroid()
    GLU          = U.select_atoms('resname {} and not altloc B and segid A'.format(am_1))
    ASP          = U.select_atoms('resname {} and not altloc B and segid A'.format(am_2))

    # Get resids for the chosen aminoacid and associate the proper set of coordinates  
    # Glu dictionary
    d_glu    = build_residues_dict(GLU.positions, GLU.resids, centroid_at)
    keys_glu = list(d_glu.keys())

    # Asp dictionary
    d_asp    = build_residues_dict(ASP.positions, ASP.resids, centroid_at)
    keys_asp = list(d_asp.keys())

    # Evaluate the centroid of each residue
    centroid_glu = eval_centroid(d_glu)
    centroid_asp = eval_centroid(d_asp)

    # Evaluate the distance between the centroid of the first and the second residue (find which one are close)
    # glu_close_to_asp : glta
    glta      = eval_distance(centroid_glu, centroid_asp)
    # find which glu are closer then thr Angstrom to a certain asp
    glta_bool = np.array(glta < thr_glu_asp) 

    # Dictionary arg:[glu closer]
    # Close_to_ASP: cta
    cta_id, cta_dt = order_residue_dict(glta_bool, glta, keys_asp, keys_glu)

    # (Find which residue 1 are close to each other) and close to the asp
    # glu_close_to_glu : gltg
    gltg      = eval_distance(centroid_glu, centroid_glu)
    gltg_bool = np.array(np.logical_and( gltg > thr_glu_glu[0], gltg < thr_glu_glu[1]) )# find which glu are closer then 10 A to a certain asp

    # Dictionary glu:[glu closer]
    # Close_to_GLU: ctg
    ctg_id, ctg_dt = order_residue_dict(gltg_bool, gltg, keys_glu, keys_glu)

    # Dictionary resid1-resid2:distance(resid1-resid2)
    #g_dt = build_dist_res(d_dt=ctg_dt,d_id=ctg_id)
    a_dt = build_dist_res(d_dt=cta_dt,d_id=cta_id)

    # Remove those residues which have an ids close to each other
    ctg_id = remove_close_id(ctg_id, ctg_dt, thr_clos_id)
    cta_id = remove_close_id(cta_id, cta_dt, thr_clos_id)

    if ctg_id and cta_id : 
        score_tot, score_list, score_dt = score_function(cta_id, ctg_id, a_dt)
        
        return score_tot, score_list, score_dt
    else:
        return None, None, None



def print_results(score_tot, score_dt, filename, outfile):
    """
        Function which takes as an input a dictionary
        and print it, with his corresponding pdb file in a 
        formatted way one the file outfile
        
        Inputs:
            score_tot : dictionary results
            score_dt  : dictionary results
            filename  : pdb file
            outfile   : file where to write
            
        Returns
            none
            
    """
    
    with open(outfile, 'a')  as f:
        f.write("---------------- file {} ---------------- \n".format(filename))
        for k, v in score_dt.items():
            if len(v) == 3 and np.logical_and((v >0.5).all(), (v<8.5).all()): 
                if  8  in np.round(v,0) and 1 in np.round(v,0) and 3 in np.round(v,0):
                    f.write("ASP " + str(k) + " GLU  " +  str(score_tot[k]) + " Dist " + str(v) + "\n")
                    
                    
def walk_QUADDEATH_PDB(initial_params, outfile='output_search.log'):
    """
        Function which walks in the pdb data base and evaluate
        the search procedure for the quadrilateral residues
        
        Inputs:
            initial_params : dictionary, set of initial parameters
            outfile : file where to write the output 
        
        Returns
            None
    """
    os.chdir(initial_params['pdb_path'])
    
    out_path = initial_params['out_path']
    outfile  = out_path + outfile

    start_time = timeit.default_timer()
    now = datetime.datetime.now()

    with open(outfile, 'w') as f:
        f.write("Start time {} \n".format(str(now.strftime("%Y-%m-%d %H:%M"))))
        
    for dirname, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            file = os.path.join(dirname, filename)
            if '.ent.gz' in file:
                try:  
                    U         = mda.Universe(file) # Define an universe
                except Exception as e:
                    with open(outfile,'a') as f:
                        f.write("---------------- file {} ---------------- \n".format(file))
                        f.write("Exception creating the universe raised for file {}. \n Error {} \n".format(file,e))
                try:
                    score_tot, score_list, score_dt  = search_quad_procedure(U, initial_params)
                except Exception as e:
                    print("Exception {}".format(e))
                    print("Error in the evalaution of the search procedure, file {}".format(file))
                if score_tot: print_results(score_tot,score_dt, file,outfile)
            score_tot = None
            U         = None
                    
                
                        
    end_time = timeit.default_timer()
    now      = datetime.datetime.now()
    with open(outfile,'a') as f:
        f.write("Elapsed time: {:.2f} s \n".format(end_time - start_time))
        f.write("End time {} \n".format(str(now.strftime("%Y-%m-%d %H:%M"))))
        