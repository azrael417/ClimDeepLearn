module unload python/2.7-anaconda-4.4
module load teca

python generate_teca_npy_files.py --dataset HAPPI15  --output_dir $SCRATCH/gb_helper/HAPPI15/label_1/ --label_version label_1

module unload teca
module load python/2.7-anaconda-4.4

python create_dataset/make_pandas_table.py --numpy_dir $SCRATCH/gb_helper/HAPPI15/label_1/ --output_dir $SCRATCH/gb_helper/HAPPI15/label_1/

python make_teca_subtables.py --table_path $SCRATCH/gb_helper/HAPPI15/label_1/teca_labels.csv --output_dir  $SCRATCH/gb_helper/HAPPI15/label_1/subtables/