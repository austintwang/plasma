import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import traceback

pal = sns.color_palette()
COLORMAP = {
	"full": pal[1],
	"indep": pal[3],
	"ase": pal[2],
	"acav": pal[5],
	"eqtl": pal[0],
	"cav": pal[7],
	"rasq": pal[6],
	"fmb": pal[8],
}
NAMEMAP = {
	"full": "PLASMA-JC",
	"indep": "PLASMA-J",
	"ase": "PLASMA-AS",
	"acav": "CAVIAR-ASE",
	"eqtl": "QTL-Only",
	"cav": "CAVIAR",
	"rasq": "RASQUAL+",
	"fmb": "FINEMAP",
}

def load_data(data_dir, test_name):
	# print(os.listdir(data_dir)) ####
	filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if i.endswith(".pickle")]
	filenames = filenames[:50] ####
	data_list = []
	for i in filenames:
		print(i) ####
		with open(i, "rb") as data_file:
			data = pickle.load(data_file)
		data_list.extend(data)

	data_df = pd.DataFrame.from_records(data_list)
	# print(data_df.columns.values) ####
	return data_df

def make_distplot(
		df,
		var, 
		model_flavors,
		model_names, 
		model_colors,
		title, 
		result_path,
		num_snps
	):

	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(4,4)})
	for m in model_flavors:
		try:
			# model_data = np.sum(df.loc[df["model"] == m, [var]].to_numpy(), axis=1)
			model_data = df.loc[df["Model"] == m, [var]].to_numpy().flatten()
			sns.distplot(
				model_data,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False},
				label=model_names[m],
				color=model_colors[m]
			)
		except Exception:
			print(traceback.format_exc())
			pass

	plt.xlim(0, num_snps)
	plt.legend(title="Model")
	plt.xlabel(var)
	plt.ylabel("Density")
	plt.title(title)
	plt.savefig(result_path, bbox_inches='tight')
	plt.clf()

def make_avg_lineplot(
		df,
		var, 
		model_flavors,
		model_names, 
		model_colors,
		title, 
		result_path,
		num_snps
	):
	inclusions_dict = {
		"Number of Selected Markers": [],
		var+" Rate": [],
		"Model": []
	}
	for m in model_flavors:
		try:
			inclusion_data = np.vstack(df.loc[df["Model"] == m, var].to_numpy())
			inclusion_agg = list(np.mean(inclusion_data, axis=0))
			# print(inclusion_data) ####
			# print(inclusion_agg) ####
			# print(len(inclusion_agg)) ####
			inclusions_dict["Number of Selected Markers"].extend(list(range(1, num_snps+1)))
			inclusions_dict[var+" Rate"].extend(inclusion_agg)
			inclusions_dict["Model"].extend(num_snps * [model_names[m]])
		except Exception:
			print(traceback.format_exc())
			pass

	inclusions_df = pd.DataFrame(inclusions_dict)
	# print(inclusions_df) ####

	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(4,4)})

	palette = [model_colors[m] for m in model_flavors]
	names = [model_names[m] for m in model_flavors]
	sns.lineplot(
		x="Number of Selected Markers", 
		y=var+" Rate", 
		hue="Model", 
		data=inclusions_df, 
		hue_order=names, 
		palette=palette
	)
	plt.xlim(0., num_snps)
	plt.title(title)
	plt.savefig(result_path, bbox_inches='tight')
	plt.clf()

def make_thresh_barplot(
		df,
		var, 
		model_flavors,
		model_names, 
		threshs,
		title, 
		result_path,
		num_snps
	):
	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(4,4)})
	palette = sns.cubehelix_palette(len(threshs))

	for i, t in enumerate(reverse(threshs)):
		estimator = lambda x: np.mean((x <= t).astype(int))
		# print(df[var]) ####
		# print(df[var].dtype) ####
		chart = sns.barplot(
			x=var, 
			y="Model", 
			data=df, 
			label=t, 
			order=model_flavors, 
			color=palette[-i-1], 
			estimator=estimator,
			ci=None
		)
		chart.set_yticklabels([model_names[m] for m in model_flavors])

	plt.title(title)
	plt.savefig(result_path, bbox_inches='tight')
	plt.clf()

def make_heatmap(
		df,
		var_row, 
		var_col, 
		response, 
		model_name, 
		title, 
		result_path, 
		aggfunc="mean",
		fmt='.2g'
	):
	heat_data = pd.pivot_table(
		df, 
		values=response, 
		index=var_row, 
		columns=var_col, 
		aggfunc=aggfunc
	)

	sns.heatmap(heat_data, annot=True, fmt=fmt, square=True)
	plt.title(title)
	plt.savefig(result_path, bbox_inches='tight')
	plt.clf()

def write_stats(
		df,
		var, 
		model_flavors,
		model_names, 
		threshs,
		result_path,
	):

	means = []
	medians = []
	stds = []
	tres = [[] for _ in range(len(threshs))]

	for m in model_flavors:
		lines = []
		# model_arr = np.array([i for i in np.array(df.loc[df["model"] == m, [var]].to_numpy())])
		# model_data = np.sum(model_arr, axis=1)
		model_data = df.loc[df["Model"] == m, var].to_numpy()
		# print(model_data) ####
		means.append(str(np.mean(model_data)))
		medians.append(str(np.median(model_data)))
		stds.append(str(np.std(model_data)))
		for i, t in enumerate(threshs):
			under = tres[i].append(str(np.mean((model_data <= t).astype(int))))

	header = "\t" + "\t".join(model_names[m] for m in model_flavors) + "\n"

	lines = []
	lines.append(header)
	lines.append("Mean\t{0}\n".format("\t".join(means)))
	lines.append("Median\t{0}\n".format("\t".join(medians)))
	lines.append("Std Dev\t{0}\n".format("\t".join(stds)))

	lines.append("\nThresholds:\n")
	lines.append(header)
	for i, t in enumerate(tres):
		lines.append("{0}\t{1}\n".format(threshs[i], "\t".join(t)))

	with open(result_path, "w") as result_file:
		result_file.writelines(lines)

def write_stats_simple(
		df,
		var, 
		model_flavors,
		model_names, 
		result_path,
	):
	lines = []
	for m in model_flavors:
		model_data = df.loc[df["Model"] == m, [var]].to_numpy()
		mean = np.mean(model_data)
		line = "{0}:\t{1}\n".format(model_names[m], mean)
		lines.append(line)

	with open(result_path, "w") as result_file:
		result_file.writelines(lines)

def interpret_mainfig(
		data_dir_base, 
		std_al_dev, 
		titles,
		model_flavors,
		model_flavors_cred,
		threshs,
		num_snps,
		res_dir_base
	):
	data_dir = os.path.join(data_dir_base, "mainfig")
	df = load_data(data_dir, "mainfig")

	res_dir = os.path.join(res_dir_base, "mainfig")
	if not os.path.exists(res_dir):
		os.makedirs(res_dir)

	var_cred = "Credible Set Size"
	var_inc = "Inclusion"

	for i, s in enumerate(std_al_dev):
		df_res = df.loc[
			(df["std_al_dev"] == s)
			& (df["complete"] == True)
		]
		df_res.rename(
			columns={
				"causal_set_size": var_cred,
				"inclusion": var_inc,
				"model": "Model",
			}, 
			inplace=True
		)

		title = titles[i]

		result_path = os.path.join(res_dir, "recall_s_{0}.txt".format(s))
		write_stats_simple(
			df_res,
			"recall", 
			model_flavors,
			NAMEMAP, 
			result_path,
		)

		result_path = os.path.join(res_dir, "stats_s_{0}.txt".format(s))
		write_stats(
			df_res,
			var_cred, 
			model_flavors,
			NAMEMAP, 
			threshs,
			result_path,
		)

		result_path = os.path.join(res_dir, "sets_s_{0}.svg".format(s))
		make_distplot(
			df_res,
			var_cred, 
			model_flavors_cred,
			NAMEMAP, 
			COLORMAP,
			title, 
			result_path,
			num_snps
		)

		result_path = os.path.join(res_dir, "inc_s_{0}.svg".format(s))
		make_avg_lineplot(
			df_res,
			var_inc, 
			model_flavors,
			NAMEMAP, 
			COLORMAP,
			title, 
			result_path,
			num_snps
		)

		result_path = os.path.join(res_dir, "thresh_s_{0}.svg".format(s))
		make_thresh_barplot(
			df_res,
			var_cred, 
			model_flavors_cred,
			NAMEMAP, 
			threshs,
			title, 
			result_path,
			num_snps
		)

if __name__ == '__main__':
	data_dir_base = "/agusevlab/awang/job_data/sim/outs/"
	res_dir_base = "/agusevlab/awang/ase_finemap_results/sim/"
	# model_flavors = set(["indep", "eqtl", "ase", "ecav"])

	std_al_dev = [0.6, 0.8]
	titles = ["Low AS Variance", "High AS Variance"]
	model_flavors = ["indep", "ase", "rasq", "acav", "eqtl", "fmb", "cav"]
	model_flavors_cred = ["indep", "ase", "acav", "eqtl", "fmb", "cav"]
	threshs = [1, 5, 20, 40, 70, 100]
	num_snps = 100
	interpret_mainfig(
		data_dir_base, 
		std_al_dev, 
		titles,
		model_flavors,
		model_flavors_cred,
		threshs,
		num_snps,
		res_dir_base
	)