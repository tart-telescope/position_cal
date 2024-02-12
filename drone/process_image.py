import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import argparse
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate the tart telescope positions from drone data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        required=False,
        default="darktable_exported/DJI_0049.jpg",
        help="Drone image to use.",
    )

    parser.add_argument(
        "--outfile",
        required=False,
        default="antenna_positions.json",
        help="Output file for antenna positions.",
    )

    parser.add_argument(
        "--template",
        required=False,
        default="template.png",
        help="Template image of a single antenna.",
    )
    
    
    ARGS = parser.parse_args()
    
    img_rgb = cv.imread(ARGS.image)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    
    template = cv.imread(ARGS.template, cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    
    cv.rectangle(img_rgb, (0,0), (200,100), (0,0,255), 2)

    w, h = template.shape[::-1]
    method = cv.TM_CCOEFF_NORMED
    i = 0
    
    antennas = []
    while True:
        res = cv.matchTemplate(img_gray,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            pt = min_loc
            best = min_val
        else:
            pt = max_loc
            best = max_val

        if best < 0.85:
            print(f"Terminating at score={best}")
            break
        
        img_gray[pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 0  # Erase find
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv.putText(img_rgb, text=f"{i}", 
                   org=(int(pt[0] + w/2), int(pt[1] + h/2)),  
                   color=(255,0,255), 
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1,
                   lineType=cv.LINE_AA )
        
        # 0,0 is top left corner
        x = pt[0] + w/2
        y = pt[1] + h/2
        
        
        x_m = x
        y_m = y
        
        
        print(f"{i}: {best} {x}, {y} {x_m}, {y_m}")
        
        antennas.append([x_m, y_m])
        i = i + 1
    cv.imwrite('res.jpg',img_rgb)
    
    antennas = np.array(antennas)
    antennas = antennas - np.mean(antennas, axis=0)
    
    result = []
    for a in antennas:
        x,y = a
        print(x, -y)
        result.append([x, -y])
        
    out_json = {"antenna_positions_pixels": result}
    with open(ARGS.outfile, "w") as fp:
        json.dump(out_json, fp, indent=4, separators=(",", ": "))

