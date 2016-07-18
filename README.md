# beta_nmf_class: group NMF with beta-divergence

Theano based GPGPU implementation of group-NMF with class and session similarity constraints.
The NMF works with beta-diveregence and multiplicative updates.

## Dependencies

beta_nmf_class need Python >= 2.7, numpy >= 10.1, Theano >= 0.8, scikit-learn >= 0.17.1, h5py >= 2.5, itertools and more_itertools

## Documentation

Documentation available at http://rserizel.github.io/groupNMF/

## Citation

If you are using this source code please consider citing the following paper: 

> R. Serizel, S. Essid, and G. Richard. “Group nonnegative matrix factorisation with speaker and session variability compensation for speaker identification”. In *Proc. of ICASSP*, pp. 5470-5474, 2016.

Bibtex
```
	@inproceedings{serizel2016group,
  	title={Group nonnegative matrix factorisation with speaker and session variability compensation for speaker identification},
  	author={Serizel, Romain and Essid, Slim and Richard, Ga{\"e}l},
  	booktitle={2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  	pages={5470--5474},
  	year={2016},
  	organization={IEEE}
	}
```

## Author

Romain Serizel, 2015 -- Present