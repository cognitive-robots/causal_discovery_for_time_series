#!/usr/bin/python3 -u

# THIS SOURCE CODE IS SUPPLIED AS IS WITHOUT WAR
# RANTY OF ANY KIND	 AND ITS AUTHOR AND THE JOURNAL OF
# ARTIFICIAL INTELLIGENCE RESEARCH JAIR AND JAIRS PUB
#LISHERS AND DISTRIBUTORS	 DISCLAIM ANY AND ALL WARRANTIES
# INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# AND ANY WARRANTIES OR NON INFRINGEMENT THE USER
# ASSUMES ALL LIABILITY AND RESPONSIBILITY FOR USE OF THIS
# SOURCE CODE	 AND NEITHER THE AUTHOR NOR JAIR	 NOR JAIRS
# PUBLISHERS AND DISTRIBUTORS	 WILL BE LIABLE FOR DAM
# AGES OF ANY KIND RESULTING FROM ITS USE Without limiting
# the generality of the foregoing	 neither the author	 nor JAIR	 nor JAIR's
# publishers and distributors	 warrant that the Source Code will be errorfree
# will operate without interruption	 or will meet the needs of the user

import time
import math
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import causal_discovery_class as cd
import networkx as nx

from tools.graph_functions import print_graph

def run_on_data(i, method, dataset, variable, files_input_name, verbose, max_time_lag, sig_level):
    save_model = True
    file_input_name = files_input_name[i]
    data = pd.read_csv(f"./data/{dataset}/{variable}/{file_input_name}", delimiter=',', index_col=0, header=0)

    variable_initial = variable[0]
    nodes = ["c0." + variable_initial, "c1." + variable_initial, "i0." + variable_initial]

    if verbose:
        print("############################## Run "+str(i)+" ##############################")
        print(file_input_name)
        print("d = " + str(data.shape[1]))
        print("T = " + str(data.shape[0]))

    gtrue = nx.DiGraph()
    gtrue.add_nodes_from(nodes)
    gtrue.add_edges_from([("c0." + variable_initial, "c1." + variable_initial)])

    start = time.time()

    errored = False
    try:
        if method == "GrangerPW":
            model = cd.GrangerPW(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "GrangerMV":
            model = cd.GrangerMV(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "TCDF":
            max_time_lag_sqrt = int(math.ceil(math.sqrt(max_time_lag)))
            model = cd.TCDF(nodes, epochs=1000,  kernel_size=max_time_lag_sqrt, dilation_coefficient=max_time_lag_sqrt, hidden_layers=1, learning_rate=0.01, sig_level=sig_level)
            model.infer_from_data(data)
        elif method == "PCMCICMIknn":
            model = cd.PCMCI(nodes, sig_level=sig_level, nlags=max_time_lag, cond_ind_test="CMIknn")
            model.infer_from_data(data)
        elif method == "PCMCIParCorr":
            model = cd.PCMCI(nodes, sig_level=sig_level, nlags=max_time_lag, cond_ind_test="ParCorr")
            model.infer_from_data(data)
        #elif method == "oCSE":
        #    model = cd.OCSE(nodes, sig_level=sig_level)
        #    model.infer_from_data(data)
        elif method == "PCTMI":
            model = cd.PCTMI(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "NBCB_pw":
            model = cd.NBCB(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "NBCB":
            model = cd.NBCB(nodes, sig_level=sig_level, nlags=max_time_lag, pairwise=False)
            model.infer_from_data(data)
        elif method == "tsFCI":
            model = cd.TsFCI(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "FCITMI":
            model = cd.FCITMI(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "VarLiNGAM":
            model = cd.VarLiNGAM(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "TiMINO":
            model = cd.TiMINO(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        #elif method == "tsKIKO":
        #    model = cd.TsKIKO(nodes, sig_level=sig_level, nlags=max_time_lag)
        #    model.infer_from_data(data)
        elif method == "Dynotears":
            model = cd.Dynotears(nodes, sig_level=sig_level, nlags=max_time_lag)
            model.infer_from_data(data)
        elif method == "NAVARMLP":
            model = cd.NAVAR(nodes, sig_level=sig_level, nlags=max_time_lag, lstm=False)
            model.infer_from_data(data)
        elif method == "NAVARLSTM":
            model = cd.NAVAR(nodes, sig_level=sig_level, nlags=max_time_lag, lstm=True)
            model.infer_from_data(data)
        elif method == "CDNOD":
            model = cd.CDNOD(nodes, sig_level=sig_level)
            model.infer_from_data(data)
        elif method == "Random":
            model = cd.RandomCausalDiscovery(nodes, edge_likelihood=0.5)
        else:
            model = None
            print("Error: method not found")
            exit(0)
    except Exception as e:
        print(e)
        errored = True

    end = time.time()

    if save_model:
        nx.write_gpickle(model.oghat, f"./experiments/graphs/{dataset}/{variable}/{max_time_lag}/{sig_level}/{method}_{i}")

    # evaluation other
    o_pres_a = model.evaluation(gtrue, evaluation_measure="other_precision_adjacent")
    o_rec_a = model.evaluation(gtrue, evaluation_measure="other_recall_adjacent")
    o_fscore_a = model.evaluation(gtrue, evaluation_measure="other_f1_adjacent")
    o_pres_o = model.evaluation(gtrue, evaluation_measure="other_precision_oriented")
    o_rec_o = model.evaluation(gtrue, evaluation_measure="other_recall_oriented")
    o_fscore_o = model.evaluation(gtrue, evaluation_measure="other_f1_oriented")

    if verbose:
        print('True Graph Other')
        print_graph(gtrue)
        print('Inferred Graph Other')
        model.print_graph()

        print("other precision adjacent: " + str(o_pres_a))
        print("other recall adjacent: " + str(o_rec_a))
        print("other f-score adjacent: " + str(o_fscore_a))
        print("other precision oriented: " + str(o_pres_o))
        print("other recall oriented: " + str(o_rec_o))
        print("other f-score oriented: " + str(o_fscore_o))

    print("Computation time: " + str(end - start))
    return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, o_pres_a, o_rec_a, o_fscore_a, o_pres_o, o_rec_o, o_fscore_o, \
            -1.0, -1.0, -1.0, (end - start)



if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Runs causal discovery on time series data and evaluates the performance of the method applied")
    arg_parser.add_argument("method", default="Dynotears")
    arg_parser.add_argument("dataset", default="lyft")
    arg_parser.add_argument("variable", default="acceleration")
    arg_parser.add_argument("--processor-count", type=int, default=1)
    arg_parser.add_argument("--verbose", action="store_true")
    arg_parser.add_argument("--max-time-lag", type=int, default=25)
    arg_parser.add_argument("--sig-level", type=float, default=0.05)
    args = arg_parser.parse_args()
    print(args)

    path_input = f"./data/{args.dataset}/{args.variable}/"
    files_input_name = [f for f in listdir(path_input) if isfile(join(path_input, f)) and not f.startswith('.')]
    results = Parallel(n_jobs=args.processor_count)(delayed(run_on_data)(i, args.method, args.dataset, args.variable, files_input_name, args.verbose, args.max_time_lag, args.sig_level)
                                             for i in range(len(files_input_name)))

    results = np.array(results).reshape(len(files_input_name), -1)
    o_pres_a_list = results[:, 6]
    o_rec_a_list = results[:, 7]
    o_fscore_a_list = results[:, 8]
    o_pres_o_list = results[:, 9]
    o_rec_o_list = results[:, 10]
    o_fscore_o_list = results[:, 11]
    comput_time_list = results[:, 15]

    with open(f"./experiments/performance_average/{args.max_time_lag}/{args.sig_level}/{args.method}_{args.dataset}_{args.variable}", "w+") as file:
        file.write("Other Precision Adjacent: \n" + str(np.mean(o_pres_a_list)) + " +- " + str(np.std(o_pres_a_list)))
        file.write("\n")
        file.write("Other Recall Adjacent: \n" + str(np.mean(o_rec_a_list)) + " +- " + str(np.std(o_rec_a_list)))
        file.write("\n")
        file.write("Other F-Score Adjacent: \n" + str(np.mean(o_fscore_a_list)) + " +- " + str(np.std(o_fscore_a_list)))
        file.write("\n")
        file.write("Other Precision Oriented: \n" + str(np.mean(o_pres_o_list)) + " +- " + str(np.std(o_pres_o_list)))
        file.write("\n")
        file.write("Other Recall Oriented: \n" + str(np.mean(o_rec_o_list)) + " +- " + str(np.std(o_rec_o_list)))
        file.write("\n")
        file.write("Other F-Score Oriented: \n" + str(np.mean(o_fscore_o_list)) + " +- " + str(np.std(o_fscore_o_list)))
        file.write("\n")

        file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    if args.verbose:
        print("####################### Final Result #######################")
        print("Other Precision Adjacent: " + str(np.mean(o_pres_a_list)) + " +- " + str(np.std(o_pres_a_list)))
        print("Other Recall Adjacent: " + str(np.mean(o_rec_a_list)) + " +- " + str(np.std(o_rec_a_list)))
        print("Other F-Score Adjacent: " + str(np.mean(o_fscore_a_list)) + " +- " + str(np.std(o_fscore_a_list)))
        print("Other Precision Oriented: " + str(np.mean(o_pres_o_list)) + " +- " + str(np.std(o_pres_o_list)))
        print("Other Recall Oriented: " + str(np.mean(o_rec_o_list)) + " +- " + str(np.std(o_rec_o_list)))
        print("Other F-Score Oriented: " + str(np.mean(o_fscore_o_list)) + " +- " + str(np.std(o_fscore_o_list)))
        print("Computational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    np.savetxt(f"./experiments/performance_detail/{args.max_time_lag}/{args.sig_level}/{args.method}_{args.dataset}_{args.variable}.csv", results[:, [6, 7, 8, 9, 10, 11, 15]],
               delimiter=';', header="o_pres_a, o_rec_a, o_fscore_a, o_pres_o, o_rec_o, o_fscore_o, computational_time")
