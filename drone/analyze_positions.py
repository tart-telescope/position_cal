import json
import argparse
import scipy

import numpy as np
from matplotlib import pyplot as plt


def get_angle(p):
    # make sure that zero degrees is at x=0, y=1 i.e., along the y axis (geographic)
    return np.degrees(np.arctan2(p[1], p[0])-np.pi/2)


def rotate(point, degrees):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    angle = np.radians(degrees)
    px, py = point

    qx = np.cos(angle) * (px) - np.sin(angle) * (py)
    qy = np.sin(angle) * (px) + np.cos(angle) * (py)
    return [qx, qy]


def r_all(v, theta):
    return np.array([rotate(p, theta) for p in v])


def translate_all(v, delta):
    return np.array([p + delta for p in v])


def r_squared(_a, _b):
    x_0, y_0 = _a
    x_p, y_p = _b
    return (x_p-x_0)**2 + (y_p-y_0)**2


def find_closest(_p, ref):
    # Find closest point in ref to p
    i_best = 0
    x_p, y_p = _p
    r_best = 9e99
    for i, orig in enumerate(ref):
        r2 = r_squared(_p, orig)
        if r2 < r_best:
            r_best = r2
            i_best = i

    return i_best, r_best


def best_permute(v, ref):
    ret = []
    for p in v:
        i_best, r_best = find_closest(p, ref)
        ret.append(i_best)
    return ret


def plot_array(ax, dr, label):
    ax.plot(dr[:,0], dr[:,1], 'x', label=label)

    x_array, y_array = dr[:,0], dr[:,1]

    i = 0
    for x,y in zip(x_array, y_array):
        circle = plt.Circle((x, y), radius=0.15/2,  color='k')
        ax.add_patch(circle,)
        ax.text(x,y, f"{i}", color="#FFFFFF", horizontalalignment='center', verticalalignment='center')
        i = i+1

def plot_pos(dr, orig, labels=["drone", "original"]):

    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,2,1, adjustable='box', aspect=1)
    plot_array(ax, dr, label=labels[0])
    ax.grid(True)
    ax.legend()

    ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=1)
    plot_array(ax2, orig, label=labels[1])
    ax2.grid(True)
    ax2.legend()
    plt.show()

if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Analyze the positions and match antennas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Original (uncalibrated) antenna positions.",
    )
    parser.add_argument(
        "--drone",
        required=False,
        default="antenna_positions.json",
        help="Input file for drone antenna positions.",
    )
    parser.add_argument(
        "--outfile",
        required=False,
        default="matched_antenna_positions.json",
        help="Output file for matched antenna positions.",
    )

    parser.add_argument(
        "--orientation", nargs='+',
        required=False,
        default=[0, 0],
        help="Array orientation [ant_num, angle].",
    )

    # Draw a line on gimp one meter long
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        required=True,
        help="Number of pixels per meter.",
    )

    ARGS = parser.parse_args()

    orient_ant, orient_degrees = ARGS.orientation
    orient_ant = int(orient_ant)
    orient_degrees = float(orient_degrees)

    print(f"Global orientation ant[{orient_ant}] = {orient_degrees} deg")

    # Opening JSON file
    with open(ARGS.drone) as f:
        drone = json.load(f)["antenna_positions_pixels"]

        drone = np.array(drone)
        mu = np.mean(drone, axis=0)
        drone = (drone - mu)/ARGS.pixels_per_meter

    with open(ARGS.original) as f:
        original = json.load(f)
        original = np.array(original)[:,0:2]
        mu = np.mean(original, axis=0)
        original = original - mu

    print(drone)
    # print(original)


    # Find the rotation that matches best (the angle that the original must be rotated to match the drone)

    def f_min(x_deg):
        ret = 0.0
        _drone_r = r_all(original, x_deg)
        for p in _drone_r:
            i_best, r_best = find_closest(p, drone)
            ret += r_best
        return ret

    # for th in np.linspace(-30,30,100):
    #     print(f"{th}, {f_min(th)}")

    ret = scipy.optimize.minimize(f_min, 40, method="BFGS", bounds=(-33,33))
    print(ret)

    angle = ret.x[0]

    original_r = r_all(original, angle)
    print(f"Angle between drone and original = {angle} degrees")
    plot_pos(dr=drone, orig=original_r)

    
    ## Find the translation that matches best to the rotated original (original_r)
    
    def t_min(_x):
        ret = 0.0
        _drone_t = translate_all(original_r,_x)
        
        for p in _drone_t:
            i_best, r_best = find_closest(p, drone)
            ret += r_best
            
        return ret

    print(t_min([0,0]))
    print(t_min([0.1,0.1]))
    print(t_min([1,1]))
    
    ret = scipy.optimize.minimize(t_min, [0,0], method="BFGS")
    print(ret)

    translate = ret.x
    original_rt = translate_all(original_r, translate)
    print(f"Translation between drone and original = {translate} m")
    plot_pos(dr=drone, orig=original_rt)

    # Now reconstruct in the original frame of reference, the drone measurements
    drone_r = r_all(drone, -angle)
    drone_rt = translate_all(drone_r, -translate)

    plot_pos(dr=drone_rt, orig=original, labels=['drone_rt', 'original'])

    # Now find the permutation that matches best the rotated original to the drone
    p_best = best_permute(original, drone_rt)

    print(p_best)
    print(original[0])
    print(drone_rt[p_best[0]])

    drone_permuted = ([list(drone_rt[i]).copy() for i in p_best])

    plot_pos(dr=np.array(drone_permuted), orig=original, labels=['drone_permuted', 'original'])

    residuals = [np.sqrt(r_squared(p, o)) for p,o in zip(drone_permuted, original)]

    # Global Orientation 
    print(f"Global orientation ant[{orient_ant}] = {orient_degrees} deg")
    ref_p = drone_permuted[orient_ant]
    actual_angle = get_angle(ref_p)
    print(f"Actual angle original[{orient_ant}] = {actual_angle} deg")

    delta_theta = orient_degrees - actual_angle
    print(f"Rotating by {delta_theta} degrees")
    drone_rotated = r_all(drone_permuted, delta_theta)

    ref_p = drone_rotated[orient_ant]
    actual_angle = get_angle(ref_p)
    print(f"Actual angle drone_rotated[{orient_ant}] = {actual_angle} deg")

    final = ([list(p) for p in drone_rotated])

    out_json = {"antenna_positions": [p + [0.0] for p in final],
                "residuals": residuals,
                "pixels_per_meter": ARGS.pixels_per_meter}

    with open(ARGS.outfile, "w") as fp:
        json.dump(out_json, fp, indent=4, separators=(",", ": "))

    final = np.array(final)
    plot_pos(dr=final, orig=original, labels=['final', 'original'])
    plt.savefig("matched_drone_pos.png")
