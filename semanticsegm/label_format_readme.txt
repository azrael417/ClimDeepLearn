Instance and semantic segmentation label storage:

Semantic segmentation: the process of classifying each label as an element of a class.  Multiple instances of the same class are not distinguished
Instance segmentation: the process of classifying each pixel as an instance of a certain class

Class_ids:
	- 0 - nothing
	- 1 - atmospheric river
	- 2 - TECA_storm

height = 768 (latitude)
width = 1152 (longitude)

How labels are stored for semantic segmentation:
	- for each image, the label is an array of shape (height, width) where each pixel is classified as 0, 1, or 2

How labels are stored for instance segmentation:
	- For a given image, there are three fields: [N, gt_boxes, gt_masks]
	- gt means ground truth
	- N = the total number of instances of any class.  N is a scalar
	- gt_boxes.shape = (N, 5) (the 5 correspond to x1, y1, x2, y2, class_id)
	- gt_masks.shape = (N, height of image, width of image)

Mask R-CNN is the model we would test for instance segmentation
Most other mainstream models (i.e. Tiramisu, which is based on DenseNet) are designed for semantic segmentation.

The final label files will be stored in the following output:

They are at the location /global/cscratch1/sd/amahesh/segmentation_labels/

	- semantic_combined_labels (in progress)
		- combined --> both ARs and teca_storms are represented
		- they will be stored for semantic segmentation (as described above)
	- instance_combined_labels (in progress)
		- combined --> both ARs and teca_storms are represented
		- they will be stored for instance segmentation (as described above)

	- semantic_storm_labels 
		- these files will just have TECA_storms (so it will be binary, 0 and 1)
	- instance_storm_labels 
		- these files will just have TECA _storms

	- semantic_ar_labels (in progress)
		- these files will just have ARs
	- instance_ar_labels (in progress)
		- these files will just have ARs

	- Location: the files are saved at /[insert_relevant_directory_from_above_options]
		- The files are saved in the following format:
		- Note: One irregularity is that the CAM5 simulation files (the .nc files) are in the format (8,768,1152).
		The 8 represents 8 time steps per day.  However, the masks are stored as (768,1152) with the appropriate time step index in the file name.
		For all files, year, month, and day are not 0-indexed.  However, time step is 0-indexed

Other relevant files
	- TECA subtables 
		- I chopped up the .bin file (generated from TECA) based on date (one file for each day TECA found a storm)
		- TECA was used to generate labels from 1996 to 2015 CAM15 data
		- These files are called teca_labels_YYYY-MM-DD-00000.csv
		- they are at /global/cscratch1/sd/amahesh/segmentation_labels/teca_subtables/
__________________________________________________________________________________________________________________________________________________________

Work that remains to be done:
	- Improve/ tinker with unsupervised binarization technique
	
	- Most storms have a very small radius (the average radius is ~1 degree).  Since each grid cell is a 2.5 degree box, that's a huge problem!


	- AR Labels
		- Since atmospheric rivers are much bigger phenomena and they are more frequent (there are about 4 or 5 in each timestep of each CAM5 image), I suspect that they will be a more interesting feature to explore, compared to teca_storms, which are smaller and rarer
		- I need some more help from Travis in order to get these labels; I've already emailed him.
		- (in the past, we used a different dataset and generated these labels from integrated vapor transport, which is a field calculated from wind and specific humidity, so some complications arose when I was working with this dataset)
			- Optional: If you are interested, here's the complication that arose.  CAM5 does not output wind as a function of pressure (just gives a static value at 850 mb), but to calculate integrated vapor transport, you need to integrate over pressure
			- The other complication is that the AR_detection algorithm needs to be a bit more refined when TMQ is the input. (When IVT is the input, the algorithm is better right out of the box)
	- After I get the AR Labels stuff sorted out, I will create combined_semantic_labels and combined_instance_labels

	- Use this source of IVT data from MERRA: 


	- No longer relevant: - Last resort for AR labels:
		- I found a place that gives pixelwise AR labels, but it is based on yet another data product.  I have already started a script to process the data, and I will need another day to finish it.
		- If all else fails, we can just try to do segmentation based on the AR labels from this dataset.  (We won't be able to use anything from TECA with this dataset.)

	- Explore this other source of IVT_data from NCEP (run the AR detection algorithm on this) http://cw3e.ucsd.edu/Publications/SIO-R1-Catalog/