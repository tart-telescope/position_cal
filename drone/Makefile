TART=rhodes
ANT_NUM=20
ANT_ANGLE=101.92

positions:
	python3 process_image.py --template ${TART}/template.png --image ${TART}/${TART}_view.jpg --outfile ${TART}/${TART}_antenna_positions.json


upload:
	echo "tart_upload_antenna_positions --api https://tart.elec.ac.nz/signal --file matched_antenna_positions.json --pw foo"

analyze:
	python3 analyze_positions.py --original ${TART}/original_antenna_positions.json \
		--drone ${TART}/${TART}_antenna_positions.json \
		--pixels-per-meter 671.37 \
		--orientation ${ANT_NUM} ${ANT_ANGLE}
