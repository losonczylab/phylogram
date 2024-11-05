from phylogram import *

hoc_file = '../example_data/example.hoc'

blank_annot_df = pd.read_csv('../example_data/annot.csv')
fig_blank, ax_blank = plot_morphology(hoc_file,annotation_df=blank_annot_df.dropna(),  markeredgewidth=1, markeredgecolor='grey')
fig2, ax2 = plot_phylogram(hoc_file, annotation_df=blank_annot_df.dropna(), linewidths=0.5, edgecolors='grey',
                           lw=1, trunk_lw=2, figsize=(4,4))

fig_blank.savefig('../example_data/example_morphology.png', bbox_inches='tight', transparent=True)
fig2.savefig('../example_data/example_phylogram.png', bbox_inches='tight', transparent=True)
