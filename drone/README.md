# Drone Position Calibration for TART antennas

Author: Tim Molteno 2024.

This is the easiest way to work out the antennas positions. A single image is captured from directly above the TART.


## Image Lens Correction

Use darktable to correct the lens for rectilinear.

## Run the antenna finder

This generates a JSON file consisting of the pixel coordinates of the antennas in the image.

    python3 process_image.py --image lens_corrected.jpg --outfile antenna_positions.json

This assumes that the lens corrected image is called 'lens_corrected.jpg'
    
    {
    "antenna_positions_pixels": [
        [
            -159.0,
            -128.41666666666674
        ],
        [
            -272.0,
            442.58333333333326
        ],
        [
            -336.0,
            -12.416666666666742
        ],
    ...

## Run the processing step

This requires an estimate of the number of pixels per meter in the image. In this case 370.9, as well as the known angle of one of the antennas. In our case antenna 5 is known to point to -10 degrees from North (angles increase counter-clockwise, so this is slightly east of North).

This also requires a file of original antenna positions (this is used to map antenna numbers to those expected by the software - this step may not be needed in future).

    python3 analyze_positions.py --original original_antenna_positions.json \
		--pixels-per-meter 370.9 \
		--orientation 5 -10
		
This will generate an array centered about the 
