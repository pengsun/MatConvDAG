This folder contains routines for DAG (training, testing, batch data generator, etc.)

* `dag_mb.m`. A thin wrapper for the DAG that arranges the associated routines. It always train and test with mini-batch, hence the name.
* `dbg_i.m`. Interface for *d*ata *b*atch *g*enerator. It defines the basic methods required by `dag_mb.m`
	* `dbg_memXd4Yd2.m`. Derived from `dbg_i.m`. It is tailored for image data (4 dimensional training instance, 2 dimensional 0/1 vector labels). All data are loaded into memory beforehand.
	* `dbg_xxx.m`. Derive your own  to implement customized data loading strategy. e.g., read the from files stored in drive or transmitted by network.