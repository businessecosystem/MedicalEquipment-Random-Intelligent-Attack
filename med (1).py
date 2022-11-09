import pandas as pd
import igraph as ig
import numpy as np
import ipyparallel
import matplotlib.pyplot as plt
import pickle
import logging
import os
from copy import deepcopy
import math
import itertools
import random
import seaborn as sns
from functools import partial
import matplotlib
#logging.basicConfig(filename='.med.log',level=logging.DEBUG,format='%(levelname)s:%(message)s',filemode='w')

max_tiers=10

def get_df(cached=True,save=True, extra_tiers=False):
    if cached:
        df = pd.read_hdf('dat/kayvan_data.h5') 
    else:
        file_name='dat/firms_tier_1_thru_10 v4 New Firms Data 2021-02-15.xlsx'

        df = pd.read_excel(file_name,sheet_name="Sheet1")
        df = df.drop('Type',axis=1)
        df = df.drop_duplicates(ignore_index=True)
        df = df.rename({'Source Name':'Source NAME','Target Name':'Target NAME',
            'Source MktCap': 'SourceSizeMktCap','Target MktCap': 'TargetSizeMktCap',
            'Source Employees Global':'SourceSizeEmployeesGlobal', 'Target Employees Global':'TargetSizeEmployeesGlobal',
            'SourceTotalRevenue':'SourceSizeRevenue','TargetTotalRevenue':'TargetSizeRevenue',
            'Source Type':'SourceType', 'Target Type':'TargetType'},axis=1)

        if extra_tiers:
            for tier in range(6,max_tiers + 1):
                print(tier)
                _,e_new = get_tier(tier,supplier_only=True)
                e_new['Tier']=tier
                f_old = df[df.Tier == tier-1].Source.unique()
                e_new = e_new[e_new.Target.map(lambda x: x in f_old)]
                df=df.append(e_new,ignore_index=True)
        df.drop(list(df.filter(regex='Unnamed')),axis=1,inplace=True)

        # resolve NaNs for better typing
        for col in ['Source Country', 'Target Country', 'Source NAME', 'Target NAME', 'Source Industry', 'Target Industry', 'SourceType', 'TargetType']:
            df[col] = df[col].astype(str)
        for col in ['SourceSizeMktCap', 'TargetSizeMktCap', 'SourceSizeRevenue', 'TargetSizeRevenue', 'SourceSizeEmployeesGlobal', 'TargetSizeEmployeesGlobal']:
            df.loc[df[col] == '(Invalid Identifier)',col] = math.nan
            df[col]=df[col].astype(float)

        if save:
            df.to_hdf('dat/kayvan_data.h5',key='df')

    return df

def get_tier(tier=6, supplier_only=False):

    # Standardize columns
    if tier == 6:
        fname = 'Backward Tier 6 v2 cleaned up.xlsx'
        to_drop = ['Source(Tier 5)','Target (Tier 4)','Type','Q.Control','Suppliers (Tier 6)','Supplier Relationship (Tier 6)']
        target_name = 'Source(Tier 5)-Becomes Target of "Tier 6 Supplier"'
    elif tier == 7:
        fname = 'tier_6_edges_all_supplier_types v3 Tier-7.xlsx'
        to_drop = ['Q.Control', 'Supplier Relationship (Tier 6)']
        target_name = 'Source(6th Tier)-becomes target for 7th tier'
    elif tier == 8:
        fname = 'tier_7_edges_8th tier added v2 duplicate source removed.xlsx'
        to_drop = ['Qcontrol', 'Supplier Relationship (Tier 6)']
        target_name = 'Source'
    elif tier == 9:
        fname = 'tier_8_firms v2 9th tier added.xlsx'
        to_drop = ['Qcontrol', 'Supplier Relationship (Tier 6)']
        target_name = 'Source'
    elif tier == 10:
        fname = 'tier_9_edges v2 Tier 10 added.xlsx'
        to_drop = ['Supplier Relationship (Tier 6)']
        target_name = 'Source'

    # Load
    df = pd.read_excel('dat/' + fname, sheet_name='Sheet1', engine='openpyxl') # Need to pip install openpyxl. The default engine now doesn't support .xlsx, only .xls
    df.drop(to_drop, axis=1,inplace=True)
    df.rename(columns={target_name:'Target'},inplace=True)

    # Get edges
    edge_df = pd.DataFrame(columns=['Source', 'Target'])
    for i in range(1,351):
        dfi=df[['Target',i,str(i)+'.1']].dropna().rename(columns = {i : 'Source', str(i) + '.1' : 'Type'})
        if supplier_only:
            dfi=dfi[dfi.Type=='Supplier'].drop('Type',axis=1)
        edge_df = edge_df.append(dfi,ignore_index=True)
    edge_df.drop_duplicates(inplace=True,ignore_index=True)

    # Get firms
    firm_df = pd.DataFrame(dict(ID=edge_df['Source']))
    firm_df = firm_df.append(pd.DataFrame(dict(ID=edge_df['Target'])))
    firm_df.drop_duplicates(inplace=True,ignore_index=True)

    return firm_df,edge_df

def get_firm_df(df=None):
    if df is None:
        df = get_df()

    df = df.rename({'Source Name':'Source NAME','Target Name':'Target NAME',
        'Source MktCap': 'SourceSizeMktCap','Target MktCap': 'TargetSizeMktCap',
        'Source Employees Global':'SourceSizeEmployeesGlobal', 'Target Employees Global':'TargetSizeEmployeesGlobal',
        'Source Type':'SourceType', 'Target Type':'TargetType'},axis=1)

    firm_df = pd.DataFrame(dict(
        ID=df['Source'],
        name=df['Source NAME'],
        industry=df['Source Industry'],
        country=df['Source Country'],
        market_cap=df['SourceSizeMktCap'],
        Employees=df['SourceSizeEmployeesGlobal'],
        private=df['SourceType']))
    firm_df = firm_df.append(pd.DataFrame(dict(
        ID=df['Target'],
        name=df['Target NAME'],
        industry=df['Target Industry'],
        country=df['Target Country'],
        market_cap=df['TargetSizeMktCap'],
        Employees=df['TargetSizeEmployeesGlobal'],
        private=df['TargetType'])))

    for col in ['country', 'industry']:
        firm_df[col] = firm_df[col].astype(str) # make NaNs into string nan
    firm_df.private = firm_df.private == 'Private Company'
    firm_df = firm_df.drop_duplicates(ignore_index=True)
    firm_df.set_index('ID',inplace=True)
    firm_df['country-industry'] = firm_df['country'] + ' '  + firm_df['industry']
    firm_df.loc[firm_df['Employees'] == '(Invalid Identifier)','Employees'] = math.nan

    firm_df['Tier'] = df.groupby('Source').Tier.min()
    firm_df.loc[firm_df.Tier.isna(),'Tier'] = 0
    firm_df.Tier = firm_df.Tier.astype(int)

    return firm_df

def get_edge_df(df=None):
    if df is None:
        df = get_df()
    return df[['Source','Target','Tier']]

def get_shortcut_edges(G):
    fdf = get_firm_df()
    fdf = pd.concat([fdf, pd.DataFrame(dict(tier=G.vs['tier'],clean_tier=G.vs['clean_tier']),index=G.vs['name'])],axis=1)
    df = get_df()
    df['Source tier'] = df.Source.map(lambda x : fdf.loc[x]['tier'])
    df['Target tier'] = df.Target.map(lambda x : fdf.loc[x]['tier'])

    return df[df['Source tier'] > df['Target tier'] + 1]

def edges():
    df=get_edge_df()
    e,i = pd.factorize(df.Source.append(df.Target,ignore_index=True)) # not guaranteed to respect order of appearance, but it does in practice
    
    m = e.size//2
    
    ee=np.zeros((m,2),dtype=int)
    ee[:,0]=e[:m]
    ee[:,1]=e[m:]
    e=ee
    return e,i

def k_core(G,k=1,fname="k_core.gml"):
    # next step -- k-core
    c=np.array(G.coreness())
    core_ind = list(np.where(c>k)[0])
    G=G.induced_subgraph(core_ind)
    save(G,fname)
    return G, core_ind

def get_med_suppliers(G):
    return list({x.target_vertex for x in G.es(tier = 1)})

def directed_igraph(*,
        giant=True,
        no_software=True,
        include_private=True,
        cut_low_terminal_suppliers=True,
        reduced_density=False,
        clean_tiers=True,
        overwrite_old_tiers=True):

    df = get_df()
    firm_df = get_firm_df(df)
    edge_df = get_edge_df(df)
    G = ig.Graph(directed=True)
    G.add_vertices(firm_df.index)
    G.vs['firm name']=firm_df['name']
    for attr in ['industry', 'country', 'country-industry','market_cap','Employees','private']:
        G.vs[attr]=firm_df[attr]
    G.vs['tier']=firm_df['Tier']

    G.add_edges(edge_df[['Source','Target']].itertuples(index=False))

    G.es['tier'] = edge_df.Tier
    G.simplify(loops=False, combine_edges='min') # use min to keep smaller tier value.

    if no_software:
        G=G.induced_subgraph(
                [x.index for x in 
                    G.vs(lambda x : x['industry'] not in 
                        ['Application Software', 'IT Consulting and Other Services', 'Systems Software', 'Advertising', 'Movies and Entertainment', 'Interactive Home Entertainment'])])

    if cut_low_terminal_suppliers:
        med_suppliers = get_med_suppliers(G) # this is not the same as when we did clean_tiers!
        t = [get_terminal_suppliers(i,G) for i in med_suppliers]
        to_delete = [m for m,tt in zip(med_suppliers,t) if len(tt) < 31] # 31 because there is a jump in len(tt) here
        G.delete_vertices(to_delete)

    if clean_tiers: # remove nodes not tracing back to tier 0
        med_suppliers = get_med_suppliers(G)
        to_keep = med_suppliers
        curr_tier = to_keep
        G.vs['clean_tier'] = [0]*G.vcount()
        #tier=0
        #while len(curr_tier)>0:
        for tier in range(0,max_tiers):

            # get next tier (as a list with duplicates)
            next_tier = []
            for v in curr_tier:
                next_tier += v.predecessors()

            #  the next current tier will be the new nodes we just found
            curr_tier = set(next_tier).difference(to_keep)
            for v in curr_tier:
                v['clean_tier'] = tier + 1

            # add next tier to the list of nodes to keep
            to_keep = set(to_keep).union(curr_tier)
            #tier += 1

        if overwrite_old_tiers:
            G.vs['tier'] = G.vs['clean_tier']
            for e in G.es:
                e['tier'] = e.target_vertex['tier'] + 1

        G = G.induced_subgraph(to_keep)
    else:
        G.vs['tier_set'] = [{e['tier'] for e in n.out_edges()}
                                for n in G.vs]
        G.vs['tier'] = [min(x['tier_set'],default=0) for x in G.vs]

    if giant:
        G=G.components(mode='WEAK').giant() # this turns out to be a no-op if you do clean-tiers with 10 tiers

    for v in get_med_suppliers(G):
        v['tier']=0
        try:
            v['tier_set'].add(0)
        except:
            pass

    if reduced_density:
        print('Reducing density')
        G=random_thinning_factory(G)(reduced_density) # .2 works. .1 might have no med suppliers

    G.vs['id'] = list(range(G.vcount())) # helps when passing to subgraphs
    med_suppliers = get_med_suppliers(G)
    G.vs['is_medical'] = [i in med_suppliers for i in G.vs]
    G.reversed = False

    return G

def reverse(G):
    tier = dict(tier=G.es['tier'])
    edges = [tuple(reversed(e.tuple)) for e in G.es]
    G.delete_edges(None)
    G.add_edges(edges,tier)
    G.reversed = not G.reversed

def get_terminal_suppliers(i,G):
    if type(i) is ig.Vertex:
        i = i.index

    try:
        G.vs['id']
    except:
        G.vs['id'] = list(range(G.vcount())) # helps when passing to subgraphs


    d = G.bfs(i,mode='IN')
    d = d[0][:d[1][-1]] # remove trailing zeros
    d = G.induced_subgraph(d)

    sccs = d.clusters()
    terminal_components = sccs.cluster_graph().vs(_indegree_eq = 0)
    sccs = list(sccs)
    terminal_nodes = [sccs[i.index] for i in terminal_components]
    terminal_nodes = itertools.chain(*terminal_nodes)
    did = d.vs['id']
    terminal_nodes = {did[i] for i in terminal_nodes}

    return terminal_nodes

# All I need now is a way to efficiently get t_thin (indices of t in G_thin)
def is_reachable(t_thin,i_thin,G_thin):
    return np.array(G_thin.shortest_paths(i_thin,t_thin)) < np.inf

# TODO: make this faster:
#   Change G to the quotient graph (once in the calling function, passed in as an optional argument) G.components().cluster_graph()
#   Then since it is a DAG maybe there is a smarter way to do this
#   Try using is_reachable above instead
def get_u(i_thick,G_thin,med_suppliers_thin=None,direction='IN'):
    if type(i_thick) is ig.Vertex:
        i_thick = i_thick.index 

    try:
        i_thin = med_suppliers_thin[i_thick] if med_suppliers_thin else G_thin.vs.find(id=i_thick).index
    except: # the node we want has been deleted
        return set()

    u = G_thin.bfs(i_thin,mode=direction)
    u = u[0][:u[1][-1]] # remove trailing zeros

    ids = G_thin.vs['id']
    return {ids[i] for i in u} 

def is_increasing(l):
    if len(l) < 2:
        return True
    for i in range(len(l)-1):
        if l[i] > l[i+1]:
            return False
    return True

# i is an integer, the index of the node in G
def some_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    if u & t: # set intersection
        return True
    return False
some_terminal_suppliers_reachable.description='Some end suppliers reachable'
some_terminal_suppliers_reachable.type=bool

def all_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    return t.issubset(u)
all_terminal_suppliers_reachable.description='All end suppliers reachable'

def percent_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    return len(t & u)/len(t)
percent_terminal_suppliers_reachable.description='Avg. percent end suppliers reachable'
percent_terminal_suppliers_reachable.type = float

def all_terminal_suppliers_exist(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)

    return t.issubset(G_thin.vs['id'])
all_terminal_suppliers_exist.description='All end suppliers surviving'

def is_subgraph(G_thin,G):

    # check nodes
    for i in G_thin.vs['id']:
        if i not in G.vs['id']:
            return False

    # check edges
    for e in G_thin.es:
        s=e.source_vertex['id']
        t=e.target_vertex['id']
        sv = G.vs(id = s)[0]
        tv = G.vs(id = t)[0]
        if not tv in sv.successors():
            return False

    return True

def random_thinning_factory(G):
    firm_rands = np.random.random(G.vcount())

    uniques = dict()
    perm = dict()
    for failure_scale in ['country', 'industry', 'country-industry']:
        uniques[failure_scale] = list(set(G.vs[failure_scale]))
        perm[failure_scale] = uniques[failure_scale]
        random.shuffle(perm[failure_scale])

    def attack(rho,failure_scale='firm'):
        if failure_scale == 'firm':
            return G.induced_subgraph((firm_rands <= rho).nonzero()[0].tolist())
        else:
            keep_uniques = perm[failure_scale][:round(rho*len(uniques[failure_scale]))]
            return G.induced_subgraph(G.vs(lambda x : x[failure_scale] in keep_uniques))
    attack.description = 'Random'

    return attack
random_thinning_factory.description = 'Random'

def get_sorted_attr_inds(G,attr):

    sorted_attr_inds=dict()
    sorted_attr_inds['firm'] = sorted(range(G.vcount()), key=G.vs[attr].__getitem__)
    for failure_scale in ['country', 'industry', 'country-industry']:
        sorted_attr_inds[failure_scale]  = sorted(set(G.vs[failure_scale ]), key=lambda x : sum(G.vs(lambda v : v[failure_scale] == x)[attr]))
    return sorted_attr_inds

def target_by_attribute(G,attr):

    sorted_attr_inds = get_sorted_attr_inds(G,attr)

    def targeted(r,failure_scale='firm'):
        to_keep = sorted_attr_inds[failure_scale][:int( len(sorted_attr_inds[failure_scale]) * r )]
        if failure_scale == 'firm':
            return G.induced_subgraph(to_keep)
        else:
            return G.induced_subgraph(G.vs(lambda x : str(x[failure_scale]) in to_keep))

    targeted.description = attr
            
    return targeted

def get_employee_attack(G):
    try:
        G.vs['Employees_imputed']
    except:
        G.vs['Employees_imputed'] = [math.isnan(x) for x in G.vs['Employees']]
    size_dist_private = np.array([x['Employees'] for x in G.vs if x['private'] and not x['Employees_imputed']])
    imputed_size = np.random.choice(size_dist_private,len(G.vs(Employees_imputed_eq = True)))
    for v,s in zip(G.vs(Employees_imputed_eq = True),imputed_size):
        v['Employees'] = s
    return target_by_attribute(G,'Employees')
get_employee_attack.description = 'Employees'

def get_degree_attack(G):
    G.vs['degree'] = G.degree(range(G.vcount()))
    return target_by_attribute(G,'degree')
get_degree_attack.description = 'Degree'

def get_pagerank_attack(G,transpose=True):

    attrname = 'Pagerank of transpose' if transpose else 'Pagerank'
    try:
        G[attrname]
    except:
        if transpose:
            reverse(G)
            pr = G.pagerank()
            reverse(G)
        else:
            pr=G.pagerank()
        G.vs[attrname]=pr

    return target_by_attribute(G,attrname)
get_pagerank_attack.description='Pagerank of transpose'

def get_pagerank_attack_no_transpose(G):
    return get_pagerank_attack(G,transpose=False)
get_pagerank_attack_no_transpose.description='Pagerank'

def get_null_attack(G):
    def null_attack(r,failure_scale='firm'):
        return G
    return null_attack
get_null_attack.description='Null'#-targeted'

def impute_industry(G):
    try:
        G['industry_imputed']
    except:
        G.vs['industry_imputed'] = [x == 'nan' for x in G.vs['industry']]

    industry_dist = np.array([x['industry'] for x in G.vs if not x['industry_imputed']])
    imputed_industry = np.random.choice(industry_dist,len(G.vs(industry_imputed_eq = True)),replace=True)
    for v,s in zip(G.vs(industry_imputed_eq = True),imputed_industry):
        v['industry'] = s

has_ipyparallel = True
try:
    dv = ipyparallel.Client()[:] # This should be global (or a singleton) to avoid an error with too many files open https://github.com/ipython/ipython/issues/6039
    dv.block=False
    dv.use_dill()
except:
    has_ipyparallel = False
    print("Loading without ipyparallel support")

callbacks = [
some_terminal_suppliers_reachable,
percent_terminal_suppliers_reachable,
]

def failure_reachability_single(r,G,med_suppliers=False,ts=False,failure_scale='firm',callbacks=callbacks,targeted=False):

    if not med_suppliers:
        med_suppliers=get_med_suppliers(G)
    if not ts:
        ts = [get_terminal_suppliers(i,G) for i in med_suppliers]
    if not targeted:
        targeted=random_thinning_factory(G)

    G_thin = targeted(r,failure_scale=failure_scale)
    med_suppliers_thin = {i_thin['id']:i_thin.index for i_thin in G_thin.vs if i_thin['id'] in med_suppliers}

    res = dict()
    us = [get_u(i,G_thin, med_suppliers_thin) for i in med_suppliers]
    for cb in callbacks:
        sample = [cb(med_suppliers,G,G_thin,t,u) for i,t,u in zip(med_suppliers,ts,us)]
        res[cb.description] = np.mean(sample)
    res['Failure scale']=failure_scale
    res['Attack type']=targeted.description
    return res

def failure_reachability_sweep(G,rho=np.linspace(.3,1,71),med_suppliers=False,ts=False,failure_scale='firm',callbacks=callbacks,targeted_factory=random_thinning_factory,parallel=False):

    if failure_scale == 'industry':
        G = deepcopy(G)
        impute_industry(G)

    if not med_suppliers:
        med_suppliers=[i.index for i in get_med_suppliers(G)]
    if not ts:
        ts = [get_terminal_suppliers(i,G) for i in med_suppliers]

    avgs = []
    if parallel:
        avgs = dv.map(failure_reachability_single,
            rho,*list(zip(*[[G,med_suppliers,ts,failure_scale,callbacks,targeted_factory(G)]]*len(rho))))
    else:
        targeted=targeted_factory(G)
        for r in rho:
            print(r)
            avgs.append(failure_reachability_single(r,G,med_suppliers,ts,failure_scale=failure_scale,callbacks=callbacks,targeted=targeted))
    
    avgs = [pd.DataFrame(a,index=[0]) for a in avgs]
    avgs = pd.concat(avgs,ignore_index=True)
    rho_name = "Percent " + get_plural(failure_scale) + " remaining"
    avgs[rho_name] = rho
    cols = list(avgs.columns)
    avgs = avgs[cols[-1:] + cols[:-1]]

    return avgs

def failure_plot(avgs,plot_title='Medical supply chain resilience under random firm failures',save_only=False,filename=None):

    rho = avgs.columns[0]
    ax=[]
    ax = [sns.lineplot(x=rho,y=col,label=col,data=avgs, errorbar=('pi',95)) for col in avgs.columns]
    ax[0].set(xlabel=rho,
            ylabel='Percent of firms',
            title=plot_title)
    plt.legend()

    if save_only:
        plt.savefig(filename)

def attack_compare_plot(
        avgs=None,
        fname='temp.svg',
        save=False,
        failure_scale='firm',
        tiers=max_tiers,
        software_included=False,
        metric=percent_terminal_suppliers_reachable.description,
        rho_scale=np.linspace(.3,1,71)):

    rho = "Percent " + get_plural(failure_scale) + " remaining"

    data=avgs[
        (avgs['Failure scale'] == failure_scale) &
        (avgs[rho] <= rho_scale[-1]) &
        (avgs[rho] >= rho_scale[0])]
    ax=sns.lineplot(x=rho,
            y=metric,
            data=data,
            hue='Attack type',
            errorbar=('pi',95),
            legend='full',estimator=np.median)
    plt.plot([rho_scale[0],rho_scale[-1]],
            [rho_scale[0],rho_scale[-1]],
            color=sns.color_palette()[data['Attack type'].unique().size],label=rho)
    ax.set(title = failure_scale.capitalize() + ' failures')
    plt.legend()

    if save:
        os.makedirs(os.path.dirname(fname),exist_ok=True)
        plt.savefig(fname)

    return ax

def clean_prefix(prefix):
    if prefix != '' and prefix[-1] != '/':
        prefix += '/'
    return prefix

def failure_reachability(G,
        rho=np.linspace(.3,1,71),
        plot=True,
        save_only=False,
        repeats=1,
        failure_scale='firm',
        use_cached_t=False,
        targeted_factory=random_thinning_factory,
        parallel='auto',
        callbacks=callbacks,
        G_has_no_software_flag=None,
        prefix='',
        med_suppliers=None):

    if parallel == 'auto' or parallel == True:
        parallel = 'repeat' if repeats > 1 else 'rho'

    if med_suppliers is None:
        med_suppliers = [i.index for i in get_med_suppliers(G)]

    if use_cached_t:
        t=[]
        with open('.cached_t','rb') as f: t=pickle.load(f)
    else:
        t = [get_terminal_suppliers(i,G) for i in med_suppliers]
        with open('.cached_t','wb') as f: pickle.dump(t,f)

    args = [[G,rho,med_suppliers,t,failure_scale,callbacks,targeted_factory]] * repeats # Beware here that the copy here is very shallow

    if parallel == 'repeat':
        avgs = dv.map(failure_reachability_sweep,
                *list(zip(*args)))
    elif parallel == 'rho':
        avgs = [failure_reachability_sweep(*args[0],parallel=True)]
    else:
        avgs = [failure_reachability_sweep(*args[0]) for _ in range(repeats)]
    avgs = pd.concat(avgs,ignore_index=True)

    if plot:
        plot_title = targeted_factory.description.capitalize() + ' '\
                + failure_scale + ' failures'\
                + ((' excluding software firms' if G_has_no_software_flag else ' including software firms') if G_has_no_software_flag is not None else '')
        prefix=clean_prefix(prefix)
        fname = failure_scale\
                + '_' + targeted_factory.description.replace(' ','_').lower()\
                + '_range_' + str(rho[0]) + '_' + str(rho[-1])\
                + '_repeats_' + str(repeats)\
                + (('software_excluded' if G_has_no_software_flag else 'software_included') if G_has_no_software_flag is not None else '')
        if save_only:
            os.makedirs(os.path.dirname(prefix + 'dat/'  + fname),exist_ok=True)
            with open(prefix + 'dat/' + fname+'.pickle',mode='wb') as f:
                pickle.dump(avgs,f)
        os.makedirs(os.path.dirname(prefix + 'im/'  + fname),exist_ok=True)
        failure_plot(avgs[avgs.columns[:-2]],
            plot_title=plot_title,
            save_only=save_only,
            filename=prefix + 'im/'+fname+'.svg')

    return avgs

def get_plural(x):
    if x=='firm': 
        return 'firms'
    elif x=='country': 
        return 'countries'
    elif x=='industry': 
        return 'industries'
    elif x=='country-industry':
        return 'country-industries'
    else:
        raise NotImplementedError

def reduce_tiers(G,tiers): 
    G.delete_edges(G.es(tier_ge = tiers+1)) # This can delete some edges even if tier=max_tier, since there can be edges of tier max_tier+1
    G.delete_vertices(G.vs(tier_ge = tiers+1))
    G.vs['id'] = list(range(G.vcount()))
    for attr in ['Pagerank', 'Pagerank of transpose', 'Employees_imputed', 'Industry_imputed']:
        try:
            del G.vs[attr]
        except:
            pass

def temp(G, 
        to_reverse,
        to_copy):
    if to_copy:
        G = deepcopy(G)

    if not to_reverse:
        return np.array(G.pagerank())
    else:
        reverse(G)
        pr = np.array(G.pagerank())
        reverse(G)
        return pr

def compare_tiers(G, 
        rho=np.linspace(.3,1,71), 
        repeats=24, 
        plot=True,
        save=True, 
        attack = random_thinning_factory, 
        failure_scale='firm',
        tier_range = range(1,max_tiers+1),
        prefix='',
        parallel='auto'):
    G = deepcopy(G)
    res = pd.DataFrame()
    for tiers in reversed(tier_range):
        print(tiers)
        reduce_tiers(G,tiers)
        res_tier = failure_reachability(
                    G,
                    rho=rho,
                    plot=False,
                    callbacks=(percent_terminal_suppliers_reachable,),
                    repeats=repeats,
                    targeted_factory = attack,
                    failure_scale = failure_scale,
                    parallel=parallel)
        res_tier['Tier count'] = tiers
        res=res.append(res_tier,ignore_index=True)
        with open(prefix + 'temp.pickle',mode='wb') as f: # save in case there is a crash
            pickle.dump(res,f)

    fname = 'compare_tiers/' + failure_scale + '/' + attack.description.replace(' ','_').lower()
    os.makedirs(prefix + 'dat/'+os.path.dirname(fname),exist_ok=True)
    res.to_hdf(prefix + 'dat/' + fname + '.h5', key='res')

    if plot:
        rho = "Percent " + get_plural(failure_scale) + " remaining"
        ax = sns.lineplot(
                        x=rho,
                        y=percent_terminal_suppliers_reachable.description,
                        data=res,
                        hue='Tier count',
                        errorbar=('pi',95),
                        legend='full')
        ax.set(title= attack.description.capitalize() + ' failures')
        if save:
            os.makedirs(prefix + 'im/'+os.path.dirname(fname),exist_ok=True)
            plt.savefig(prefix + 'im/'+fname+'.svg')
    return res

def no_china_us_reachability(G,include_taiwan_hong_kong=False,prefix='.'):
    print("Removing all US-China supply chain links")

    # Get medical suppliers
    firms = G.vs
    firms['is_med'] = [False] * G.vcount()
    for x in G.es(tier = 1): x.target_vertex['is_med'] = True

    # define china
    china=['China']
    if include_taiwan_hong_kong:
        print("Including Taiwan and Hong Kong in China")
        china += ['Hong Kong','Taiwan']
    else:
        print("Excluding Taiwan and Hong Kong from China")

    # find chinese/us firms
    is_ch = lambda x : type(x['country'] == str) and (x['country'] in china)
    is_us = lambda x : x['country'] == 'United States'

    # thin graph
    G_thin = deepcopy(G)
    G_thin.delete_edges(
            G_thin.es.select(
                _between = [
                    G_thin.vs(is_ch), 
                    G_thin.vs(is_us)]))
    #print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]
    u = [get_u(i.index,G_thin) for i in med_suppliers]

    reachable = pd.DataFrame([[cb(i.index,G,G_thin,tt,uu) for i,tt,uu in zip(med_suppliers,t,u)] for cb in callbacks],index=[cb.description for cb in callbacks]).transpose()
    reachable['country'] = G.vs([i.index for i in med_suppliers])['country']
    for cb in callbacks:
        reachable[cb.description] = reachable[cb.description].astype(cb.type)

    by_country = reachable.groupby('country').mean().reindex(['United States', 'China', 'Taiwan', 'Hong Kong'])

    by_country.to_excel(prefix + '/dat/no_us_china'+ ('_incl_taiwan_hk' if include_taiwan_hong_kong else '') + '.xlsx')

    return by_country

def close_all_borders(G,prefix='.'):
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(lambda e : e.source_vertex['country'] != e.target_vertex['country']))

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]
    u = [get_u(i.index,G_thin) for i in med_suppliers]

    reachable = pd.DataFrame([[cb(i.index,G,G_thin,tt,uu) for i,tt,uu in zip(med_suppliers,t,u)] for cb in callbacks],
            index=[cb.description for cb in callbacks]).transpose()
    reachable['country'] = G.vs([i.index for i in med_suppliers])['country']
    for cb in callbacks:
        reachable[cb.description] = reachable[cb.description].astype(cb.type)

    by_country = reachable.groupby('country').mean()

    by_country.to_excel(prefix+'/dat/close_all_borders.xlsx')

    return reachable

def industry_deletion_effects(G,prefix='.'):

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]

    res = dict()
    for ind in set(G.vs['industry']):
        if type(ind) is not float:
            G_thin = deepcopy(G)
            G_thin.delete_vertices(G.vs(industry=ind))
            u = [get_u(i.index,G_thin) for i in med_suppliers]
            res[ind] = [np.mean([cb(None,G,G_thin,tt,uu) for tt,uu in zip(t,u)]) for cb in callbacks]

    res = pd.DataFrame(res,index=[cb.description for cb in callbacks]).transpose()
    res.to_excel(prefix + '/dat/industry_deletion_effects.xlsx')

    return res

def run_all_simulations(
        G=None,
        attacks=None,
        repeats=None,
        giant=True,
        rho_scales=None,
        software_compare=False,
        scales_simulations=True,
        tiers_simulations=True,
        borders=True,
        tiers=range(1,max_tiers+1),
        write_mode='w',
        prefix=''):

    if G is None:
        G = directed_igraph(giant=giant)

    old_backend = matplotlib.backends.backend
    matplotlib.use('Agg') # non-interactive

    full_rho = np.linspace(.3,1,71)
    failure_scales = ['firm','country','industry','country-industry']
    if attacks is None:
        attacks = [random_thinning_factory, partial(get_pagerank_attack,transpose=True), partial(get_pagerank_attack, transpose=False), get_employee_attack]
        attacks[1].description = 'Pagerank of transpose'
        attacks[2].description = 'Pagerank'

    max_repeats=100
    if repeats=='min':
        max_repeats=6

#    if (repeats is None) or (repeats == 'min'):
#        repeats = dict([(random_thinning_factory,max_repeats),
#            (attacks[1],1),
#            (attacks[2],1),
#            (get_employee_attack,(6 if max_repeats==6 else 24))])
    if (repeats is None) or (repeats == 'min'):
        repeats = dict()
        for attack in attacks:
            for failure_scale in failure_scales:
                repeats[attack,failure_scale] = 1
                if failure_scale == 'industry':
                    repeats[attack,failure_scale] = min(max_repeats,24)
                if (attack == random_thinning_factory) or (attack == get_employee_attack):
                    repeats[attack,failure_scale] = max_repeats


    if rho_scales is None:
        rho_scales = [full_rho, np.linspace(.9,1,101)]#, np.linspace(.99,1,101),np.linspace(.999,1,101),np.linspace(.9999,1,101)]

    med_suppliers = [i.index for i in get_med_suppliers(G)]


    prefix=clean_prefix(prefix)
    os.makedirs(prefix + 'dat/',exist_ok=True)
    resfile = prefix + 'dat/all_results.h5'

    to_return = []
    if scales_simulations:
        res = pd.DataFrame()
        for rho in rho_scales:
            for failure_scale in failure_scales:
                for attack in attacks:
                    print(failure_scale + ' ' + (attack.description if attack else 'random') + ' scale ' + str(rho[0]) + ' ' + str(rho[-1]))
                    plt.clf()
                    avgs=failure_reachability(G,
                            rho=rho, 
                            targeted_factory=attack, 
                            plot=False,
                            repeats=repeats[(attack,failure_scale)], 
                            failure_scale=failure_scale,
                            med_suppliers=med_suppliers,
                            prefix=prefix)
                    res=res.append(avgs,ignore_index=True)
        res.to_hdf(resfile,key='scales',mode=write_mode)
        to_return.append(res)

        print('plotting')
        for failure_scale in failure_scales:
            res_temp = res[res['Failure scale'] == failure_scale]
            for rho in rho_scales:
                for metric in callbacks:
                    print(failure_scale + ' ' + str(rho[0]) + ' ' + metric.description)
                    plt.clf()
                    attack_compare_plot(res_temp,
                            failure_scale=failure_scale,
                            rho_scale=rho,
                            fname = prefix + 'im/attack_compare_' + str(rho[0]) + '_' + str(rho[-1]) + '/'\
                                    + metric.description.replace(' ','_').replace('.','').lower()\
                                    + '/' + failure_scale + '.svg',
                            save=True,
                            metric=metric.description)

    if tiers_simulations:
        res = pd.DataFrame()
        for failure_scale in failure_scales:
            for attack in attacks:
                print('compare tiers ' + attack.description.lower() + ' ' + failure_scale)
                plt.clf()
                res = res.append(compare_tiers(G,
                    rho=full_rho,
                    repeats=repeats[(attack,failure_scale)],
                    plot='save',
                    attack=attack,
                    failure_scale=failure_scale,
                    tier_range=tiers,
                    prefix=prefix),
                    ignore_index=True)
        res.to_hdf(resfile,key='tiers',
                mode=('a' if scales_simulations else write_mode))
        to_return.append(res)


    if software_compare:
        res = pd.DataFrame()
        graphs = (directed_igraph(no_software = True, giant=giant), 
                directed_igraph(no_software = False, giant=giant))
        for G, inclusive  in zip(graphs, (False, True)):
            for attack in attacks:
                print('Software ' + ('included' if inclusive else 'excluded') + ' ' + (attack.description if attack else 'random'))
                plt.clf()
                avgs=failure_reachability(G,
                        rho=full_rho, 
                        targeted_factory=attack, 
                        save_only=True,
                        repeats=repeats[(attack,failure_scale)], 
                        failure_scale='firm',
                        G_has_no_software_flag = (not inclusive),
                        prefix=prefix + 'software_compare')
                avgs['Software included'] = inclusive
                res=res.append(avgs,ignore_index=True)
        res.to_hdf(resfile,key='software',
                mode=('a' if scales_simulations or tiers_simulations else write_mode))
        res['Software included']=res['Software included'].astype(bool)
        to_return.append(res)

    if borders:
        print('no_china_us_reachability')
        no_china_us_reachability(G,include_taiwan_hong_kong=True,prefix=prefix)
        print('no_china_us_reachability')
        no_china_us_reachability(G,include_taiwan_hong_kong=False)
        print('close_all_borders')
        close_all_borders(G,prefix=prefix)
        print('industry_deletion_effects')
        industry_deletion_effects(G,prefix=prefix)
    
    matplotlib.use(old_backend) # non-interactive ('Qt5Agg' for me)

    return tuple(to_return)

def tree_graph_test():
    tree_depth = 5
    tree_count = 20
    repeats=12
    degree=4
    tol = .05

    N=int((degree**(tree_depth+1) -1) / (tree_depth - 1))

    Gs=[]
    for i in range(tree_count):
        G=ig.Graph.Tree(N,degree,"in")
        for scale in ['country', 'industry', 'country-industry']:
            G.vs[scale] = ['foo']*N
        G.vs['name'] = list(range(i*N,(i+1)*N))
        Gs.append(G)
    G = ig.operators.union(Gs)

    med_suppliers = [i*N for i in range(tree_count)]

    res=failure_reachability(G,med_suppliers=med_suppliers,repeats=repeats)

    gb = res.groupby('Percent firms remaining')
    observed = gb.mean()['Avg. percent end suppliers reachable']
    rho = gb.first().index
    expected = rho**(tree_depth+1)
    plt.plot(rho,expected)

    abs_err = np.max(np.abs(observed - expected))
    if abs_err < tol:
        print("passed with error " + str(abs_err))
    else:
        print("failed with error " + str(abs_err))

    return res

#x=pd.DataFrame(index=failure_scales,columns=[attack.description for attack in attacks])
#for attack in attacks:
#    for scale in failure_scales:
#        x[attack.description][scale] = med.required_tiers(res,attack,scale)

def required_tiers(res,attack,scale):
    # Assume res is already filtered to a specific attack and scale

    res=res[(res['Failure scale']==scale) & (res['Attack type'] == attack.description)]
    rho = "Percent " + get_plural(scale) + " remaining"

    means=[]
    for tier_count in range(1,max_tiers+1):
        means.append(res[res['Tier count'] == tier_count].groupby(rho)['Avg. percent end suppliers reachable'].mean())

    maxes = np.zeros(max_tiers)
    for tier_count in range(1,max_tiers+1):
        maxes[tier_count-1] = np.max(np.abs(means[tier_count-1] - means[-1]))

    tol = .05
    return np.nonzero(maxes<tol)[0][0]+1

def breakdown_thresholds(res,tol=.2):

    t = pd.DataFrame()
    for attack in [random_thinning_factory, get_pagerank_attack, get_pagerank_attack_no_transpose, get_employee_attack]:
        for scale in res['Failure scale'].unique():
            x=res[(res['Failure scale']==scale) & (res['Attack type'] == attack.description)].groupby('Percent ' + get_plural(scale) + ' remaining').mean()['Avg. percent end suppliers reachable']
            try:
                val = x.index[np.max(np.nonzero((x<tol).values))]
            except:
                val = .29
            t=t.append({'Scale': scale, 'Attack':attack.description, 'val':val},ignore_index=True) # get largest post-threshold density
    return t

def er_threshold(G,rho=np.linspace(0,1,101),repeats=10):
    for rr in range(repeats):
        print(rr)
        a = random_thinning_factory(G)
        impute_industry(G)
        failure_scales = ['firm','country','industry','country-industry']
        thr = pd.DataFrame()
        for failure_scale in failure_scales:
            print(failure_scale)
            d = pd.Series(index=rho,name='mean_degree')
            for r in rho:
                Gt = a(r,failure_scale=failure_scale)
                d[r] = np.mean(Gt.degree())
            thr=thr.append({'Scale':failure_scale, 'val':d.index[np.nonzero(d.values>1)[0][0]]},ignore_index=True)
    thr=thr.groupby('Scale')['val'].median()
    return thr

import powerlaw
def powerlaw_threshold_random(G):
    gamma = powerlaw.Fit(np.bincount(G.degree())).alpha # exponent
    kmax=max(G.degree())
    kmin=min(G.degree())

    kappa = (2-gamma)/(3-gamma) * kmax # https://en.wikipedia.org/wiki/Robustness_of_complex_networks#Critical_threshold_for_random_failures
    fc = 1 - 1/(kappa-1)

    return 1-fc

def powerlaw_threshold_targeted(G):
    kmin=min(G.degree())
    gamma = powerlaw.Fit(np.bincount(G.degree())).alpha # exponent

    def f(x):
        return x**((2-gamma)/(1-gamma)) - 2 - (2-gamma)/(3-gamma) * kmin * (x**((3-gamma)/(1-gamma)) - 1)

    from scipy.optimize import root_scalar
    return root_scalar(f,bracket=[0,1]) # This doesn't work, since alpha is about 1.4<2. So the network is exceedingly fragile against degree-based attacks (at least in terms of giant connected components)

# ER attack is the same as ER failure

