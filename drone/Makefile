positions:
	python3 process_image.py --image lens_corrected.jpg --outfile antenna_positions.json


upload:
	echo "tart_upload_antenna_positions --api https://tart.elec.ac.nz/signal --file matched_antenna_positions.json --pw foo"

analyze:
	python3 analyze_positions.py --original original_antenna_positions.json \
		--pixels-per-meter 370.9 \
		--orientation 5 -10
