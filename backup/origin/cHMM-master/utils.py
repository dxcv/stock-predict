# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:28:52 2018

@author: eric.benhamou, david sabbagh, valentin melot
"""

import os.path

# Image resolution
DPI = 300


def delete_all_png_file():
    for item in os.listdir('.'):
        if item.endswith(".png"):
            try:
                os.remove(os.path.join(os.getcwd(), item))
            except Exception as e:
                print(e.message)
    return


def save_figure(plt, prefix, suffix, lgd=None):
    filename = '{}_figure_{}.png'.format(prefix, suffix)
    count = 1
    while os.path.isfile(filename):
        count = count + 1
        filename = '{}_figure_{}({}).png'.format(prefix, suffix, count)
    if lgd is not None:
        plt.savefig(filename, bbox_extra_artists=(
            lgd,), bbox_inches='tight', dpi=DPI)
    else:
        plt.savefig(filename, dpi=DPI)
    plt.show(block=False)
    print('saved figure as \'{}\''.format(filename))
    return
