inventorexploration
====
This is the code accompanying the paper:
Jeff Alstott, Giorgio Triulzi, Bowen Yan, Jianxi Luo. (2017). “Inventors' Explorations Across Technology Domains.” 
The manuscript is in this repository `here`__, on SSRN `here`__ and is in press at `Design Science`__.

__ https://github.com/jeffalstott/inventorexploration/raw/master/manuscript/Alstott_et_al_PDF.pdf
__ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2936709
__ http://www.designsciencejournal.org/

How to Reproduce the Study
====
The code base is organized as a set of `IPython notebooks`__, which are also duplicated as simple Python ``.py`` script files. To reproduce the full study, the only thing you should need to touch directly is the notebook `Manuscript_Code`__ , which walks through all the steps of:

1. calculating the relatedness between technology domains from patent data, by creating randomized versions of history and comparing the empirical data to it
2. creating a predictive model of inventors' movements, using as predictors relatedness, popularity, and other factors.
3. creating figures for the `manuscript`__, the source code for which is also contained in this repository.

__ http://ipython.org/notebook.html
__ https://github.com/jeffalstott/inventorexploration/blob/master/src/Manuscript_Code.ipynb
__ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2936709

The data files involved are too large to host on Github (>100MB), and so they are hosted on Zenodo `here__`. Just download the contents to 'data/' and you should be good to go. 

__ https://zenodo.org/record/1035458

How to Do Your Own Analysis
====
Reproducing the full study would require significant computational resources (see below). As such, the data download also includes final versions of the data, which will allow to you to recreate just the final analyses described in the manuscript. This would also be a sensible starting place for doing your own analysis, answering new questions with the same data.

Randomization with a cluster
====
This pipeline involves creating thousands of randomized versions of the historical patent data. In order to do this, we employ a computational cluster running the `PBS`__ job scheduling system. Running this code currently assumes you have one of those. If you are lucky enough to be from the future, maybe you have a big enough machine that you can simply create and analyze thousands of randomized versions of the historical patent data using a simple ``for`` loop. We don’t yet support that.

__ https://en.wikipedia.org/wiki/Portable_Batch_System


Dependencies
====
- Python 3.x
- `powerlaw`__
- `seaborn`__
- `pyBiRewire`__
- `cmdstan`__
- the standard scientific computing Python stack, which we recommend setting up by simply using the `Anaconda Python distributon`__. Relevant packages include:
- - numpy
- - scipy
- - matplotlib

__ https://github.com/jeffalstott/powerlaw
__ http://stanford.edu/~mwaskom/software/seaborn/
__ https://github.com/andreagobbi/pyBiRewire
__ http://mc-stan.org/interfaces/cmdstan
__ http://docs.continuum.io/anaconda/index

Original Data Files
====
- citing_cited.csv
- PATENT_US_CLASS_SUBCLASSES_1975_2011.csv
- pid_issdate_ipc.csv
- disamb_data_ipc_citations_2.csv
- pnts_multiple_ipcs_76_06_valid_ipc.csv
- patent_ipc_1976_2010.

Contact
====
Please contact the authors if you have questions/comments/concerns/stories:
gtriulzi at mit
alstott at mit
