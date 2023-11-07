# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:05:49 2022

@author: vija
"""

import matplotlib.pyplot as plt
import networkx as nx


def visualize_ehub(info, params, 
                   draw_adj = True, 
                   draw_graph = True, 
                   save_figures = True):
    
    if draw_adj == True:
        fig1, ax1 = plt.subplots()
        ax1.imshow(info.adj, cmap="Blues")
        ax1.set_xticks(range(info.num_nodes))
        ax1.set_yticks(range(info.num_nodes))
        ax1.set_title("Adjacency matrix", fontsize=12)
        plt.show()
        if save_figures == True:
            fig1.savefig('output/adj.png')
    G = nx.from_pandas_edgelist(params['branches'],'node_in','node_out', 
                                create_using=nx.DiGraph()) 
    if draw_graph == True: 
        # https://ansegura7.github.io/Algorithms/graphs/Graphs.html
        # https://networkx.org/documentation/stable/reference/generators.html
        # G = nx.DiGraph(G)
        fig2, ax2 = plt.subplots()
        nx.draw(G, 
                pos = nx.spring_layout(G), # nx.planar_layout(G)
                with_labels=True)
        plt.tight_layout()
        if save_figures == True:
            fig2.savefig('output/graph.png')      
        
    return G
            
            
def visualize_operation(data,
                        results,
                        temp_nodes     = [1,2],
                        power_branches = [1,2],
                        save_figures = False):
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig.set_size_inches(12, 9)
    ax1.grid()
    ax2.grid()
    fs = 10
    for n in temp_nodes: 
        node_label = 'Node'+str(n)
        ax1.plot(results['temp'][:,n], lw = 1.0, label=node_label)    
    ax1.set_ylabel('Temperature (°C)',fontsize = fs)
    ax1.legend(fontsize = fs)
    for b in power_branches: 
        branch_label = 'Branch'+str(b)
        ax2.plot(results['power'][:,b], lw = 1.0, label=branch_label)
    ax2.set_ylabel('Power (kW)',fontsize = fs)
    ax2.legend(fontsize = fs)
    
    # ax1.set_ylim(0,2)
    # ax1.set_xlim(-0.5,10.5)
    # ax1.xaxis.set_major_locator(plt.MaxNLocator(11))
    # ax1.set_xticklabels(parameters_labels, fontsize = fs)
    # formatter = DateFormatter('%d/%m %H:%M')
    # plt.gcf().axes[1].xaxis.set_major_formatter(formatter)
    # ax2.set_ylabel(r"$\theta^{i,avg}$ [°C]", fontsize = fs)
    # ax3.set_xlim(0,cs.calset.maxloops)
    # ax3.set_ylim(0,1.8)
    # ax3.set_ylabel('RMSE [K]', fontsize = fs)
    # ax3.set_xlabel('PSO iterations', fontsize = fs)
    # scat1 = ax1.scatter(np.arange(11), np.ones(11), s = 30) 
    # #line2, = ax1.scatter(x2, y2, 'b:', lw = 1.5, label=r"$\theta^{i,meas}$")
    # 
    # line4, = ax2.plot(cs.history.index, cs.history['Ti_meas_avg'], 'k--', lw = 1.5, label=r"$\theta^{i,opt}$")
    # line5, = ax3.plot(np.arange(len(rmse_train_hist)), rmse_test_hist, 'o-', lw = 1.0, label = 'Train')
    # line6, = ax3.plot(np.arange(len(rmse_test_hist)), rmse_test_hist, 'o:', lw = 1.0, label='Test') 
    # ax1.tick_params(axis='both', which='both', labelsize=ls)
    # ax2.tick_params(axis='both', which='major', labelsize=ls)
    # ax3.tick_params(axis='both', which='major', labelsize=ls)
    # ax3.set_xticklabels(iteration_labels, fontsize = fs)
    # ax2.legend(fontsize = fs)
    # ax3.legend(fontsize = fs)
    # x0 = np.ones((cs.calset.population,cs.calset.dims))
    # textbox = ax3.text(0.1,0.1, "", bbox={'alpha':0.4, 'pad': 3}, 
    #                    transform=ax3.transAxes, ha='left')
    # textbox.set_text('RMSE = ' + str(rmse_train_nom)[:5] + ' K')
    plt.show()
    if save_figures == True:
        fig.savefig('output/operation.png') 
    return


# fig1 = plt.gcf()
# plt.show()
# plt.draw()
# fig1.savefig('tessstttyyy.png', dpi=100)