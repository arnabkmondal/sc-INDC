for ds in '10X_PBMC' 'Adam' 'mouse_bladder_cell' 'mouse_ES_cell' 'Muraro' 'Quake_10x_Bladder' 'Quake_10x_Limb_Muscle' 'Quake_10x_Spleen' 'Quake_Smart-seq2_Diaphragm' 'Quake_Smart-seq2_Limb_Muscle' 'Quake_Smart-seq2_Lung' 'Quake_Smart-seq2_Trachea' 'Romanov' 'worm_neuron_cell' 'Young';
  do
    python calculate_dist.py --dataset $ds --nb_genes 3000
    for i in 0 1 2;
      do
        python noisy_signal_performance_main.py --ds $ds --expt run1 --nb_genes 3000 --z_dim 16
      done
  done