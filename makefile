
#>>>> https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate 
# <<<<

all: install_headless_test install_data test_headless

install_test:
	conda env remove -n megnet_test
	mamba create --override-channels --channel=conda-forge --name=megnet_test mne pip pytest -y
	($(CONDA_ACTIVATE) megnet_test ; pip install -e .['testing']  )

install_data:
	tmp_dir=$(pwd)
	($(CONDA_ACTIVATE) datalad ; git clone ${MEG_DATA_SERVER}:test_data /data/NIGHTLY_TESTDATA/test_data ; cd /data/NIGHTLY_TESTDATA/test_data;  datalad get /data/NIGHTLY_TESTDATA/test_data/* ) 
	cd ${tmp_dir}  #Revert out of datalad download location

install_headless_test:
	conda env remove -n megnet_test
	mamba create --override-channels --channel=conda-forge --name=megnet_test mne pip pytest "vtk>=9.2=*osmesa*" "mesalib=21.2.5" -y
	($(CONDA_ACTIVATE) megnet_test ; pip install -e .['testing'] )

install_system_requirements:
	dnf install Xvfb -y
	dnf install git git-annex -y

test:
	($(CONDA_ACTIVATE) megnet_test ; pytest -vv  )  

test_headless:
	($(CONDA_ACTIVATE) megnet_test ;  xvfb-run -a pytest -vv  )


test_iterate_fs:
	($(CONDA_ACTIVATE) megnet_test ; cd enigma_MEG; pytest -vv --report-log=./test_logfile.txt )  #xvfb-run -a pytest -s )

