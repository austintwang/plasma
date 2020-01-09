import os
import pickle
import numpy as np
import pandas as pd 
import mygene

def entrez_to_ensembl(entrez):
    mg = mygene.MyGeneInfo()
    # print(mg.getgenes(entrez[:2], fields="ensembl.gene")) ####
    gene_info = mg.getgenes(entrez, fields="ensembl.gene")
    # for i in gene_info: ####
    #     try:
    #         i["ensembl"]
    #     except Exception:
    #         print(i)
    return [i.get("ensembl", []) for i in gene_info]

def get_essential(effect_path, output_path):
    effects = pd.read_csv(effect_path, index_col=0)
    total = effects.loc["ACH-000649",:].to_numpy()
    essential = total - np.mean(effects.to_numpy(), axis=0) 
    entrez = [i.split()[1].strip("()") for i in effects.columns]
    ensembl = entrez_to_ensembl(entrez)
    essential_info = {}
    total_info = {}
    for ind, ids in enumerate(ensembl):
        try:
            essential_info[ids["gene"]] = essential[ind]
            total_info[ids["gene"]] = total[ind]
        except(AttributeError, TypeError):
            for i in ids:
                essential_info[i["gene"]] = essential[ind]
                total_info[i["gene"]] = total[ind]

    with open(output_path, "wb") as output_file:
        pickle.dump({"pref": essential_info, "total": total_info}, output_file)

if __name__ == '__main__':
    val_path = "/agusevlab/awang/job_data/validation"
    effect_path = os.path.join(val_path, "Achilles_gene_effect.csv")
    output_path = os.path.join(val_path, "essential.pickle")
    get_essential(effect_path, output_path)

