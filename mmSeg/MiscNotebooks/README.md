# Notebooks

Testing Notebooks

edgeDET:
Detection of edges in the sample images

horSig:
Generates the horizontal signal used for margin detection, using the edge images from edgeDET

load_ground_truth_csv:
Loading the ground truth from CSV mark up of the sample images

margin_threshold_test:
A script to test the threshold factor of the average value of the horSig after the ruler has been removed from consideration

margin_extraction:
The script used to extract the margins using the heuristic

Resizing4:
Resize the sample images by a factor of 4

Resizing8:
Resize the sample images by a factor of 8

Ruler_Heuristic:
Determines whether the rules is on the top/bottom or side of the sample image (No outputs).
Does the analysis of the vertical images (Peaks)

Ruler_Removed_Diff:
Does horSig for images, comparing with and without the ruler

Ruler_Threshold_Test:
Testing of different thresholds to remove the ruler

verSig4:
Generates the vertical signal used for ruler detection. Uses resized by 4x images

verSig8:
Generates the vertical signal used for ruler detection. Uses resized by 8x images
