import os
import pickle
import glob
import numpy as np
import pandas as pd 
import mygene

def entrez_to_ensembl(entrez):
    mg = mygene.MyGeneInfo()
    gene_info = mg.getgenes(entrez, fields="ensembl.gene")
    return [i["ensembl.gene"] for i in gene_info]

def get_essential(effect_path, output_path):
    effects = pd.read_csv(effect_path, index_col=0)
    essential = effects.loc["ACH-000649",:].to_numpy() - np.mean(effects.to_numpy(), axis=0) 
    entrez = [i.split[1].strip("()") for i in effects.columns]
    gene_ids = entrez_to_ensembl(entrez)

    with open(output_path, "rb") as output_file:
        pickle.dump({"ids": gene_ids, "scores": essential}, output_file)

if __name__ == '__main__':
    val_path = "/agusevlab/awang/job_data/validation"
    effect_path = os.path.join(val_path, "Achilles_gene_effect.csv")
    output_path = os.path.join(val_path, "essential.pickle")
    get_essential(effect_path, output_path)

