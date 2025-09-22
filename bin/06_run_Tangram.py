#!/usr/bin/env python3

# ----------
# Title:       06_run_Tangram.py
# Details:     Koen Rademaker (kr23@sanger.ac.uk) /
#              Elina Jin (ej10@sanger.ac.uk) / 
#              Alexander Aivazidis (alexander.aivazidis@dkfz-heidelberg.de, previously aa16@sanger.ac.uk),
#              4 November 2024
# Function:    
# Changes:
# * 0.1        Koen Rademaker
# - Implemented stand-alone script to run cell type mapping with Tangram that handles the diverse input
# arguments we require from our Xenium dataset and downstream segmentation.
# 
# * 0.2        Elina Jin
# - The layer of Anndata you want to run Tangram on can be selected by the argument input 'layer_sp' and 'layer_sc'. Note that the X layer of the saved annotated adata_sp output will become the layer you selected.
# - Running Tangram on selected marker gene set: First run rank_genes_groups on adata_sc to acquire the differentially expressed genes for each cell type cluster. The training gene set for Tangram would be selected as the overlap between the marker gene set (top 100 DEG) and the spatial gene set. Make sure that the marker gene set is stored in adata_sc.uns
# - Figure outputs:
#   1. Validation plot: from Tangram package that compares the original spatial expression profile with imputated expression profile on validation genes (overlapping genes that are not in training gene set). This plot is only produced when using marker-gene set. If running Tangram on all overlapping genes, would not have validation genes. 
#   2. prob_dist_all: plot mapping probability distribution (the highest probability for each cell, i.e. the probability associated with the assigned cell type for each cell) across all cell types. 
#   3. prob_dist_celltype: mapping probability distribution broken down by celltypes, can reflect mapping confidence for each cell type. 
# Usage:       Use BSUB script ~/GBM_nichecompas/scripts/run_step06_yyyymmdd_tangram_mode.sh
# ----------


# Import packages
import argparse
import sys
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tangram as tg
from tqdm import tqdm
import logging
import os
import pandas as pd
from sklearn.metrics import auc
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True


# Parse CLI arguments
def parse_args():
    """Parses command-line options for main()."""
    summary = 'Runs Tangram cell type mapping with labelled single-cell (or single-nucleus) \
    RNA-sequencing reference expression profiles against single-cell spatial transcriptomics (Xenium).'

    parser = argparse.ArgumentParser(description=summary)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-adata_sp',
                               required = True,
                               type = str,
                               help = 'Input spatial dataset in Anndata format.')
    requiredNamed.add_argument('-adata_sc',
                               required = True,
                               type = str,
                               help = 'Input reference single-cell (or single-nucleus) dataset in Anndata format.')
    requiredNamed.add_argument('-xenium_region_name',
                               required = True,
                               type = str,
                               help = 'Xenium region name corresponding to "adata_sp". Should match with "regionName" in \
                                      default Xenium output structure (https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs#overview)')
    requiredNamed.add_argument('-expression_type',
                               required = True,
                               type = str,
                               help = 'Specification of expression data types of spatial and single-cell inputs to Tangram, \
                                       which should be identical (e.g. raw counts or normalised counts). Value will be reflected \
                                       in Tangram output files, recommended values: "counts" or "log1pnorm".')
    requiredNamed.add_argument('-marker_genes',
                               type = str,
                               default = None,
                               help = 'Feature name of the marker genes stored in the adata_sc.uns.')
    requiredNamed.add_argument('-scRNA_name',
                               required = True,
                               type = str,
                               help = 'Name of scRNA-seq dataset to which spatial data was mapped.')
    requiredNamed.add_argument('-cell_type_label',
                               required = True,
                               type = str,
                               help = 'Cell type labels, must exist in "adata_sp" (adata_sp.obs[cell_type_label]).')
    requiredNamed.add_argument('-anndata_out',
                               required = True,
                               type = str,
                               help = 'Path to write output AnnData object to in H5AD-format.')
    requiredNamed.add_argument('-tangram_mapping_out',
                               required = True,
                               type = str,
                               help = 'Path to write the ad_map output.')
    parser.add_argument('-layer_sp',
                               type = str,
                               default = None,
                               help = 'Gene expression data (counts or log1p-norm) stored \
                                       in "adata_sp.layers" as input for Tangram.')
    parser.add_argument('-layer_sc',
                               type = str,
                               default = None,
                               help = 'Gene expression data (counts or log1p-norm) stored \
                                       in "adata_sc.layers" as input for Tangram.')
    parser.add_argument('-sub_cell_type_label',
                        type = str,
                        default = None,
                        help = 'Cell subcluster labels, must exist in "adata_sp" (adata_sp.obs[sub_cell_type_label]).')
    parser.add_argument('-log1p_normalize_spatial',
                        action='store_true',
                        help = 'Log1p-normalize Xenium data? (default: False).')
    parser.add_argument('-log1p_normalize_single_cell',
                        action='store_true',
                        help = 'Log1p-normalize single-cell RNA data? (default: False).')
    parser.add_argument('-figure_dir',
                        type = str,
                        help = 'Path to directory to store Tangram training QC plots.')
    parser.add_argument('-num_epochs',
                        default = 1000,
                        type = int,
                        help = 'Number of epochs for Tangram model training.')
    parser.add_argument('-n_marker_genes',
                        default = 100,
                        type = int,
                        help = 'Number of marker genes to be used if -marker-genes is provided.')
    parser.add_argument('-dry_run',
                        action='store_true',
                        help = 'Dry-run of the script to test all functionalities (default: False).')
    parser.add_argument('-validation_plot',
                        action='store_true',
                        help = 'Generate validation plots on test set genes? CAUTION: Can require >200 GB of memory (default: False).')
    try:    
        opts = parser.parse_args()
    except:
        sys.exit(0)
    
    return opts



def paste_top_classes(ad_map,
                      cluster_label : str,
                      probability_cutoff = 0.5):
    """Paste together highest probability cell type names that together
    make up > 50% of probability for each cell.

    Args:
        ad_map (AnnData): Cell-by-voxel structure where ad_map.X[i, j] gives the
                          probability for cell i to be in voxel j. 
        cluster_label (str): Cell type cluster label.

    Returns:
        top_class: List of length J (voxels), where element j (str) contains highest probability
                   cell type name for voxel j.
        classification : List of length J (voxels), where element j (str) contains highest probability
                         cell type names that together make up > 50% of probability for voxel j.
        n_classes : List of length J (voxels), where element j (int) denotes the number of cell types
                    stored in 'classification'.
        top_probs : List of length J (voxels), where element j (float) denotes the summed probability
                    over top cell types in voxel j.
    """
    # Normalize so rows sum to 1
    ad_map.X = (ad_map.X/np.sum(ad_map.X, axis = 0))
    order = np.argsort(-1*ad_map.X, axis = 0)
    # Initialize variables
    celltypes = ad_map.obs[cluster_label]
    n_cells = order.shape[1]
    classification = []
    n_classes = []
    top_class = []
    top_probs = []
    # Iterate over cells and assign top cell types and their probability
    for i in tqdm(range(n_cells)):
        count = 0
        o = order[count,i]
        name = celltypes[o]
        top_class += [name]
        total = ad_map.X[o,i]
        top_prob = ad_map.X[o,i]
        count += 1
        while total < probability_cutoff:
            o = order[count,i]
            name += '+' + celltypes[o]
            total += ad_map.X[o,i]
            count += 1
        classification += [name]
        n_classes += [count]
        top_probs += [top_prob]

    return top_class, classification, n_classes, top_probs


def get_most_likely_cell_types(adata_sp,
                               ad_map,
                               cluster_label : str,
                               cluster_level : str):
    """Add highest probability cell types from Tangram results to spatial transcriptomics AnnData object. 

    Args:
        adata_sp (AnnData): AnnData object with spatial cell-gene matrix.
        ad_map (AnnData): Cell-by-voxel structure where ad_map.X[i, j] gives the
                          probability for cell i to be in voxel j. 
        cluster_label (str): Cell type label from either args.cell_type_label or 
                             args.sub_cell_type_label.
        cluster_level (str): Level of cell type labels, either broad ('cell_type') or
                             subcluster ('sub_cell_type').

    Returns:
        adata_sp: Updated AnnData object for spatial transcriptomics with highest probability cell types
        from Tangram stored in adata_sp.obs (for name structure, see in 'output_obs_names').
    """    
    # Defines how output (from paste_top_classes function) should be named in adata_sp.obs
    output_obs_names = {
        'top_cluster' : {'cell_type': 'cell_type',
                         'sub_cell_type' : 'sub_cell_type'},
        'top_clusters' : {'cell_type': 'top_cell_types',
                          'sub_cell_type' : 'top_sub_cell_types'},
        'n_top_cluster' : {'cell_type': 'n_cell_types',
                           'sub_cell_type' : 'n_sub_cell_types'},
        'summed_probability' : {'cell_type': 'cell_type_probability',
                                'sub_cell_type' : 'sub_cell_type_probability'},
    }

    # Paste cell types names that make up majority of probability for each spatial cell
    top_class, classification, n_classes, top_probs = paste_top_classes(ad_map, cluster_label)
    
    # Update adata_sp object.
    adata_sp.obs[output_obs_names['top_cluster'][cluster_level]] = top_class
    adata_sp.obs[output_obs_names['top_clusters'][cluster_level]] = classification
    adata_sp.obs[output_obs_names['n_top_cluster'][cluster_level]] = n_classes
    adata_sp.obs[output_obs_names['summed_probability'][cluster_level]] = top_probs

    return adata_sp


def plot_training_scores(adata_map, bins=10, alpha=0.7):
    """
    Plots the 4-panel training diagnosis plot. Adapted from https://github.com/broadinstitute/Tangram/blob/master/tangram/plot_utils.py
    to export the figure and save to a custom path rather than only display it.

    Args:
        adata_map (AnnData):
        bins (int or string): Optional. Default is 10.
        alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

    Returns:
        fig : Matplotlib figure
    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    df = adata_map.uns["train_genes_df"]
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    #     axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df, y="train_score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_sp",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )
    plt.tight_layout()

    return fig


def plot_top_prob_all(ad_map, cluster_label : str):
    top_class, classification, n_classes, top_probs = paste_top_classes(ad_map, cluster_label)
    fig, axs = plt.subplots(2,1,figsize = (8,10))
    sns.histplot(top_probs, ax = axs[0])
    axs[0].set_title("Histogram of top cell type probability per voxel") 
    axs[0].set_ylabel("Voxel count") 
    axs[0].set_xlabel("Top cell type probability")
    
    unique_values, counts = np.unique(n_classes, return_counts=True)
    plt.bar(unique_values, counts)
    axs[1].set_title("Histogram of [number of cell types to make up 50% probability] per voxel") 
    axs[1].set_ylabel("Voxel count") 
    axs[1].set_xlabel("number of cell types to make up 50% probability")

    top_class_prob = {key: [] for key in set(top_class)}
    for i in range(len(top_class)):
        cell_type = top_class[i]
        probability = top_probs[i]
        top_class_prob[cell_type].append(probability)
    plt.tight_layout()
    plt.show()
    return fig 


def plot_top_prob_per_celltype (adata_st, cluster_label = "cell_type", probability_label = "cell_type_probability"):
    n_clusters = len(np.unique(adata_st.obs[cluster_label]))
    pal = sns.color_palette(palette='coolwarm', n_colors= n_clusters)
    
    g=sns.FacetGrid(adata_st.obs, row= cluster_label , hue= cluster_label , aspect=7.5, height=1.5, palette=pal)
    
    g.map(sns.histplot, probability_label,
          fill=True, 
          alpha=1,
          linewidth=1.5,
          bins=50,
          stat = "percent")
         
    return g.fig


# Functions for AUC plot from Tangram since these are broken in the package itself
def plot_auc(df_all_genes, test_genes=None):
    """
        Plots auc curve which is used to evaluate model performance.
    
    Args:
        df_all_genes (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp); 
        test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

    Returns:
        None
    """
    metric_dict, ((pol_xs, pol_ys), (xs, ys)) = eval_metric(df_all_genes, test_genes)
    
    fig = plt.figure(figsize = (6,5))

    plt.plot(pol_xs, pol_ys, c='r')
    sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel('score')
    plt.ylabel('spatial sparsity')
    plt.tick_params(axis='both', labelsize=8)
    plt.title('Prediction on test transcriptome')
    
    textstr = 'auc_score={}'.format(np.round(metric_dict['auc_score'], 3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment='top', bbox=props);
    return fig


def eval_metric(df_all_genes, test_genes=None):
    """
    Compute metrics on given test_genes set for evaluation
    
    Args:
        df_all_genes (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp); 
        test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

    Returns:      
        dict with values of each evaluation metric ("avg_test_score", "avg_train_score", "auc_score"), 
        tuple of auc fitted coordinates and raw coordinates(test_score vs. sparsity_sp coordinates)
    """

    # validate test_genes:
    if test_genes is not None:
        if not set(test_genes).issubset(set(df_all_genes.index.values)):
            raise ValueError(
                "the input of test_genes should be subset of genes of input dataframe"
            )
        test_genes = np.unique(test_genes)

    else:
        test_genes = list(
            set(df_all_genes[df_all_genes["is_training"] == False].index.values)
        )

    # calculate:
    test_gene_scores = df_all_genes.loc[test_genes]["score"]
    test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
    test_score_avg = test_gene_scores.mean()
    train_score_avg = df_all_genes[df_all_genes["is_training"] == True]["score"].mean()

    # sp sparsity weighted score
    test_score_sps_sp_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sp))
        / (1 - test_gene_sparsity_sp).sum()
    )

    # tm metric
    # Fit polynomial'
    xs = list(test_gene_scores)
    ys = list(test_gene_sparsity_sp)
    pol_deg = 2
    pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
    pol_xs = np.linspace(0, 1, 10)  # x linearly spaced
    pol = np.poly1d(pol_cs)  # build polynomial as function
    pol_ys = [pol(x) for x in pol_xs]  # compute polys
    
    if pol_ys[0] > 1:
        pol_ys[0] = 1

    # if real root when y = 0, add point (x, 0):
    roots = pol.r
    root = None
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
            root = roots[i]
            break

    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)       
        
    np.append(pol_xs, 1)
    np.append(pol_ys, pol(1))

    # Remove points that are out of [0,1]
    del_idx = []
    for i in range(len(pol_xs)):
        if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
            del_idx.append(i)

    pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
    pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

    # Compute area under the curve of polynomial
    auc_test_score = np.real(auc(pol_xs, pol_ys))
    metric_dict = {
        "avg_test_score": test_score_avg,
        "avg_train_score": train_score_avg,
        "sp_sparsity_score": test_score_sps_sp_g2,
        "auc_score": auc_test_score,
    }
    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))
    return metric_dict, auc_coordinates


def do_dry_run(args):
    logging.info(f'---------- DRY RUN ----------')
    logging.info(f'Script arguments: {args}')
    logging.info(f'---------- Simulating steps ----------')
    logging.info(f'Loading single-cell cell-gene matrix from : {args.adata_sc}')
    logging.info(f'Loading spatial cell-gene matrix from : {args.adata_sp}')
    if args.layer_sp is not None:
        logging.info(f'Running Tangram on adata_sp.layers[{args.layer_sp}]')
    if args.layer_sc is not None:
        logging.info(f'Running Tangram on adata_sc.layers[{args.layer_sc}]')
    if args.log1p_normalize_spatial == True:
        logging.info(f'[-log1p_normalize_spatial] Applying log1p-normalisation to spatial data ...')
    if args.log1p_normalize_single_cell == True:
        logging.info(f'[-log1p_normalize_single_cell] Applying log1p-normalisation to single-cell data ...')
    if args.marker_genes is not None: 
        logging.info(f'Selecting top {args.n_marker_genes} marker genes per reference cell type from adata_sc.uns["{args.marker_genes}"]')
        logging.info(f'Running Tangram : cluster_label={args.cell_type_label}, mode="clusters", density_prior="uniform", num_epochs={args.num_epochs}')
        logging.info(f'Parsing highest probability cell types that represent >50% for cells ...')
    if args.sub_cell_type_label is not None:
        logging.info(f'Running Tangram : cluster_label={args.sub_cell_type_label}, mode="clusters", density_prior="uniform", num_epochs={args.num_epochs}')
        logging.info(f'Parsing highest probability cell types that represent >50% for cells ...')
    _fig_path = os.path.join(
        args.figure_dir,
        '__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+'.png'
    )
    logging.info(f'Plotting training scores : {_fig_path}')
    if args.sub_cell_type_label is not None:
        _fig_path = os.path.join(
            args.figure_dir,
            '__'.join([args.xenium_region_name, args.scRNA_name, args.sub_cell_type_label])+'.png'
        )
        logging.info(f'Plotting training scores : {_fig_path}')
    if args.marker_genes != None:
        _fig_path = os.path.join(args.figure_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+ '__validation_plot.png')
        logging.info(f'Saving validation plot : {_fig_path}')
        if args.sub_cell_type_label is not None:
            _fig_path = os.path.join(args.figure_dir,'__'.join([args.xenium_region_name, args.scRNA_name, args.sub_cell_type_label])+ '__validation_plot.png')
            logging.info(f'Saving validation plot : {_fig_path}')
    _fig_path = os.path.join(args.figure_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+'__prob_dist_all.png')
    logging.info(f'Saving probability distribution plot : {_fig_path}')
    if args.sub_cell_type_label is not None:
        _fig_path = os.path.join(args.figure_dir,'__'.join([args.xenium_region_name, args.scRNA_name, args.sub_cell_type_label])+'__prob_dist_all.png')
        logging.info(f'Saving probability distribution plot : {_fig_path}')
    _fig_path = os.path.join(args.figure_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+'__prob_dist_celltype.png')
    logging.info(f'Saving cell type probability distribution plot : {_fig_path}')
    if args.sub_cell_type_label is not None:
        fig_val_celltype_path = os.path.join(args.figure_dir,'__'.join([args.xenium_region_name, args.scRNA_name, args.sub_cell_type_label])+'__prob_dist_celltype.png')
        logging.info(f'Saving cell type probability distribution plot : {fig_val_celltype_path}')
        logging.info(f'Writing results to output : {args.anndata_out}')
    # Exit script
    sys.exit(0)
    


def main():
    # Parse input arguments
    args = parse_args()
    
    # ---------------------------------------------------------------------------------------------
    # Do a dry run of Tangram
    if args.dry_run == True:
        do_dry_run(args)
    # ---------------------------------------------------------------------------------------------
    

    # ---------------------------------------------------------------------------------------------
    # Check for existence of input files
    if (not os.path.exists(args.adata_sp)):
        print("The specified input (%s) does not exist!" % args.adata_sp)
        sys.exit(0)
    if (not os.path.exists(args.adata_sc)):
        print("The specified input (%s) does not exist!" % args.adata_sc)
        sys.exit(0)
    # ---------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------
    # Create decidated output figure directory
    fig_out_dir = os.path.join(args.figure_dir,
                               args.xenium_region_name)
    os.makedirs(fig_out_dir,
                    exist_ok=True)
    # ---------------------------------------------------------------------------------------------

   
    # ---------------------------------------------------------------------------------------------
    # Read scRNA-seq cell-gene matrix
    logging.info(f'Loading single-cell cell-gene matrix from : {args.adata_sc}')
    adata_sc = sc.read_h5ad(args.adata_sc)

    # Read Xenium cell-gene matrix
    logging.info(f'Loading spatial cell-gene matrix from : {args.adata_sp}')
    adata_sp = sc.read_h5ad(args.adata_sp)
    adata_sp_var_names_original = adata_sp.var_names

    # Load gene expression data from particular Anndata object layer when provided
    if args.layer_sp != None:
        logging.info(f'Running Tangram on adata_sp.layers[{args.layer_sp}]')
        adata_sp.X = adata_sp.layers[args.layer_sp]
        logging.info(f'DEBUG: adata_sp.layers[{args.layer_sp}] summed counts per cell : {np.sum(adata_sp.X, axis=1)}')
        logging.info(f'DEBUG: adata_sp.layers[{args.layer_sp}] summed counts per gene : {np.sum(adata_sp.X, axis=0)}')
    if args.layer_sc != None:
        logging.info(f'Running Tangram on adata_sc.layers[{args.layer_sc}]')
        adata_sc.X = adata_sc.layers[args.layer_sc]
        logging.info(f'DEBUG: adata_sc.layers[{args.layer_sc}] summed counts per cell : {np.sum(adata_sc.X, axis=1)}')
        logging.info(f'DEBUG: adata_sc.layers[{args.layer_sc}] summed counts per gene : {np.sum(adata_sc.X, axis=0)}')

    # Log1p-normalise spatial data when prompted
    if args.log1p_normalize_spatial == True:
        logging.info(f'Applying log1p-normalisation to spatial data ...')
        sc.pp.normalize_total(adata_sp, inplace=True)
        sc.pp.log1p(adata_sp)

    # Log1p-normalise single-cell data when prompted
    if args.log1p_normalize_single_cell == True:
        logging.info(f'Applying log1p-normalisation to single-cell data ...')
        sc.pp.normalize_total(adata_sc, inplace=True)
        sc.pp.log1p(adata_sc)

    # Prepare AnnData objects
    logging.info(f'Preparing AnnData objects for Tangram')
    if args.marker_genes is not None: 
        logging.info(f'Selecting top {args.n_marker_genes} marker genes for each reference single-cell cell type')
        markers_df = pd.DataFrame(adata_sc.uns[args.marker_genes]["names"]).iloc[0:args.n_marker_genes, :]
        markers = list(np.unique(markers_df.melt().value.values))
        tg.pp_adatas(adata_sc,adata_sp, genes=markers)
    else:
        tg.pp_adatas(adata_sc, adata_sp, genes=None)
    logging.info(f'adata_sc.uns["training_genes"]: {adata_sc.uns["training_genes"]}')
    # ---------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------
    # Map reference single-cell cell types to Xenium
    logging.info(f'Running Tangram : cluster_label={args.cell_type_label}, mode="clusters", density_prior="uniform", num_epochs={args.num_epochs}')
    ad_map = tg.map_cells_to_space(
        adata_sc, 
        adata_sp,         
        mode = 'clusters',
        device = 'cuda',
        num_epochs = args.num_epochs,
        density_prior = 'uniform',
        cluster_label = args.cell_type_label
    )
    tg.project_cell_annotations(ad_map,
                                adata_sp,
                                annotation=args.cell_type_label)
    # Determine most likely cell type (top softmax-output from Tangrma cell type label transfer)
    logging.info(f'Parsing highest probability cell types that represent >50% for cells ...')
    adata_sp = get_most_likely_cell_types(adata_sp,
                                          ad_map,
                                          args.cell_type_label,
                                          'cell_type')
    # (Optional) Map reference single-cell subtypes to Xenium
    if args.sub_cell_type_label is not None:
        logging.info(f'Running Tangram : cluster_label={args.sub_cell_type_label}, mode="clusters", density_prior="uniform", num_epochs={args.num_epochs}')
        ad_map_sub = tg.map_cells_to_space(
            adata_sc, 
            adata_sp,         
            mode = 'clusters',
            device = 'cuda',
            num_epochs = args.num_epochs,
            density_prior = 'uniform',
            cluster_label = args.sub_cell_type_label
        )
        tg.project_cell_annotations(ad_map_sub,
                                    adata_sp,
                                    annotation=args.sub_cell_type_label)
        # Determine most likely cell subtype (top softmax-output from Tangrma cell type label transfer)
        logging.info(f'Parsing highest probability cell types that represent >50% for cells ...')
        adata_sp = get_most_likely_cell_types(adata_sp,
                                              ad_map_sub,
                                              args.sub_cell_type_label,
                                              'sub_cell_type')
    # ---------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------
    # Plot training scores for reference cell types
    fig = plot_training_scores(ad_map, bins=20, alpha=.5)
    fig_train_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+ '__training-scores.png')
    logging.info(f'Saving training scores plot : {fig_train_path}')
    fig.savefig(fig_train_path, dpi=300)

    # (Optional) Plot training scores for reference cell subtypes
    if args.sub_cell_type_label is not None:
        fig = plot_training_scores(ad_map_sub, bins=20, alpha=.5)
        fig_train_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.sub_cell_type_label])+ '__training-scores.png')
        logging.info(f'Saving training scores plot : {fig_train_path}')
        fig.savefig(fig_train_path, dpi=300)
    # ---------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------
    # Validation plot
    if args.validation_plot == True:
        if args.marker_genes != None:
            ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc, cluster_label=args.cell_type_label)
            df_all_genes = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)
            fig_val = plot_auc(df_all_genes)
            fig_val_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+ '__validation_plot.png')
            logging.info(f'Saving validation plot : {fig_val_path}')
            fig_val.savefig(fig_val_path, dpi=300)

            # (Optional) Plot validation for reference cell subtypes
            if args.sub_cell_type_label is not None:
                ad_ge = tg.project_genes(adata_map=ad_map_sub, adata_sc=adata_sc, cluster_label=args.sub_cell_type_label)
                df_all_genes = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)
                fig_val = plot_auc(df_all_genes)
                fig_val_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.sub_cell_type_label])+ '__validation_plot.png')
                fig_val.savefig(fig_val_path, dpi=300)
                logging.info(f'Saving validation plot : {fig_val_path}')
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # Plot probability distribution plot for reference cell types
    fig_val_all = plot_top_prob_all(ad_map, args.cell_type_label)
    fig_val_all_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+'__prob_dist_all.png')
    fig_val_all.savefig(fig_val_all_path, dpi=300)
    logging.info(f'Saving probability distribution plot : {fig_val_all_path}')

    # (Optional) Plot probability distribution plot for reference cell subtypes
    if args.sub_cell_type_label is not None:
        fig_val_all = plot_top_prob_all(ad_map_sub, args.sub_cell_type_label)
        fig_val_all_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.sub_cell_type_label])+'__prob_dist_all.png')
        fig_val_all.savefig(fig_val_all_path, dpi=300)
        logging.info(f'Saving probability distribution plot : {fig_val_all_path}')
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # Plot reference cell type probability distribution plot
    fig_val_celltype = plot_top_prob_per_celltype(adata_sp,
                                                  cluster_label='cell_type',
                                                  probability_label='cell_type_probability')
    fig_val_celltype_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.cell_type_label])+'__prob_dist_celltype.png')
    fig_val_celltype.savefig(fig_val_celltype_path, dpi=300)
    logging.info(f'Saving cell type probability distribution plot : {fig_val_celltype_path}')

    # (Optional) Plot reference cell subtype probability distribution plot
    if args.sub_cell_type_label is not None:
        fig_val_celltype = plot_top_prob_per_celltype(adata_sp,
                                                      cluster_label='sub_cell_type',
                                                      probability_label='sub_cell_type_probability')
        fig_val_celltype_path = os.path.join(fig_out_dir,'__'.join([args.xenium_region_name, args.expression_type, args.scRNA_name, args.sub_cell_type_label])+'__prob_dist_celltype.png')
        fig_val_celltype.savefig(fig_val_celltype_path, dpi=300)
        logging.info(f'Saving cell type probability distribution plot : {fig_val_celltype_path}')
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # Restore original gene names (Tangram lowercases them)
    adata_sp.var_names = adata_sp_var_names_original
    # Save results
    logging.info(f'Writing results to output : {args.anndata_out}')
    adata_sp.write_h5ad(args.anndata_out)
    ad_map.write_h5ad(args.tangram_mapping_out)
    # ------------------------------------------------------------------------------------------
    

if __name__ == "__main__":
    main()